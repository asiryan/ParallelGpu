using ParallelGpu;

namespace ImageDenoising;

/// <summary>Generates Gaussian noise and removes it with patch-based non-local means entirely on the GPU.</summary>
/// <remarks>Image decoding, buffer transfers, and PNG encoding belong to the caller. Device buffers
/// retain the immutable original, noisy input, and filtered output between operations.</remarks>
public sealed class GpuImageDenoiser : IDisposable
{
    /// <summary>The largest supported patch radius and resident difference-image border.</summary>
    private const int Border = 4;
    /// <summary>The borrowed CUDA or OpenCL executor.</summary>
    private readonly Gpu gpu;
    /// <summary>The original image width.</summary>
    private readonly int width;
    /// <summary>The original image height.</summary>
    private readonly int height;
    /// <summary>The number of pixels in each unpadded image.</summary>
    private readonly int pixelCount;
    /// <summary>The width of the difference image including its replicated border.</summary>
    private readonly int paddedWidth;
    /// <summary>The height of both difference and horizontal-sum images.</summary>
    private readonly int paddedHeight;
    /// <summary>The immutable RGB bytes uploaded once.</summary>
    private readonly GpuBuffer<byte> original;
    /// <summary>The byte-quantized noisy RGB colors stored as floats for distance calculations.</summary>
    private readonly GpuBuffer<DenoisingColor> noisy;
    /// <summary>The weighted color sums, replaced by the normalized filtered image after each run.</summary>
    private readonly GpuBuffer<DenoisingColor> filtered;
    /// <summary>The reusable RGB-byte conversion target.</summary>
    private readonly GpuBuffer<byte> rgbOutput;
    /// <summary>The per-offset RGB mean squared differences over the padded image.</summary>
    private readonly GpuBuffer<float> differences;
    /// <summary>The horizontal patch sums, retaining the vertical border.</summary>
    private readonly GpuBuffer<float> horizontalSums;
    /// <summary>The sum of all neighbor weights for each image pixel.</summary>
    private readonly GpuBuffer<float> weights;
    /// <summary>The exact sum of byte-quantized squared RGB errors for each pixel.</summary>
    private readonly GpuBuffer<long> squaredErrors;
    /// <summary>Whether a noisy input has been prepared.</summary>
    private bool hasNoisy;
    /// <summary>Whether a filtered output corresponds to the current noisy input.</summary>
    private bool hasFiltered;
    /// <summary>Whether the owned buffers have been released.</summary>
    private bool disposed;

    /// <summary>Uploads an RGB image and allocates persistent GPU working buffers.</summary>
    /// <param name="gpu">A GPU executor; CPU accelerators are rejected.</param>
    /// <param name="rgb">Top-to-bottom packed red, green, blue bytes.</param>
    /// <param name="width">The positive image width.</param>
    /// <param name="height">The positive image height.</param>
    public GpuImageDenoiser(Gpu gpu, byte[] rgb, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        ArgumentNullException.ThrowIfNull(rgb);
        if (!gpu.IsGpu)
            throw new ArgumentException("ImageDenoising requires a GPU; CPU processing is not supported.", nameof(gpu));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        pixelCount = checked(width * height);
        int rgbLength = checked(pixelCount * 3);
        if (rgb.Length != rgbLength)
            throw new ArgumentException("The RGB array must contain exactly three bytes per image pixel.", nameof(rgb));
        this.gpu = gpu;
        this.width = width;
        this.height = height;
        paddedWidth = checked(width + 2 * Border);
        paddedHeight = checked(height + 2 * Border);
        int differenceLength = checked(paddedWidth * paddedHeight);
        int horizontalLength = checked(width * paddedHeight);
        var allocated = new List<IDisposable>();
        try
        {
            original = gpu.Upload(rgb);
            allocated.Add(original);
            noisy = gpu.Allocate<DenoisingColor>(pixelCount);
            allocated.Add(noisy);
            filtered = gpu.Allocate<DenoisingColor>(pixelCount);
            allocated.Add(filtered);
            rgbOutput = gpu.Allocate<byte>(rgbLength);
            allocated.Add(rgbOutput);
            differences = gpu.Allocate<float>(differenceLength);
            allocated.Add(differences);
            horizontalSums = gpu.Allocate<float>(horizontalLength);
            allocated.Add(horizontalSums);
            weights = gpu.Allocate<float>(pixelCount);
            allocated.Add(weights);
            squaredErrors = gpu.Allocate<long>(pixelCount);
            allocated.Add(squaredErrors);
        }
        catch
        {
            for (int index = allocated.Count - 1; index >= 0; index--)
                allocated[index].Dispose();
            throw;
        }
    }

    /// <summary>Adds independent deterministic Gaussian noise to the original image on the GPU.</summary>
    /// <param name="sigma">The standard deviation in byte-intensity units, from zero through 100.</param>
    /// <param name="seed">The seed combined with each pixel and channel index.</param>
    /// <remarks>Values are clamped and rounded to bytes before filtering, so the saved noisy PNG
    /// contains exactly the samples used by non-local means. Repeated calls restart from the original.</remarks>
    public void AddGaussianNoise(float sigma, ulong seed)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ValidateSigma(sigma);
        GpuView<byte> input = original.DeviceView;
        GpuView<DenoisingColor> output = noisy.DeviceView;
        gpu.For(0, pixelCount, index =>
        {
            int offset = index * 3;
            output[index] = new DenoisingColor(
                Quantize(input[offset] + sigma * Gaussian(offset, seed)),
                Quantize(input[offset + 1] + sigma * Gaussian(offset + 1, seed)),
                Quantize(input[offset + 2] + sigma * Gaussian(offset + 2, seed)));
        });
        hasNoisy = true;
        hasFiltered = false;
    }

    /// <summary>Copies an already noisy input, such as a RayTracer frame, into the GPU filter source.</summary>
    public void UseInputAsNoisy()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        GpuView<byte> input = original.DeviceView;
        GpuView<DenoisingColor> output = noisy.DeviceView;
        gpu.For(0, pixelCount, index =>
        {
            int offset = index * 3;
            output[index] = new DenoisingColor(input[offset], input[offset + 1], input[offset + 2]);
        });
        hasNoisy = true;
        hasFiltered = false;
    }

    /// <summary>Filters the current noisy image using RGB patch similarities on the GPU.</summary>
    /// <param name="sigma">The known or estimated noise deviation in byte units, from zero through 100.</param>
    /// <param name="strength">The positive filter-width multiplier, at most four; h equals strength times sigma.</param>
    /// <param name="searchRadius">The neighborhood radius, from one through sixteen.</param>
    /// <param name="patchRadius">The compared square-patch radius, from zero through four.</param>
    /// <remarks>Patch pixels clamp to image edges, while search centers outside the image are skipped.
    /// Each search offset uses separable box sums instead of rereading the complete patch for every
    /// comparison. Distances average squared differences over RGB and the complete patch. Expected
    /// noise variance 2*sigma*sigma is subtracted before evaluating exp(-distance/(h*h)).</remarks>
    public void Denoise(float sigma, float strength, int searchRadius, int patchRadius)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        EnsureNoisy();
        ValidateSigma(sigma);
        if (!float.IsFinite(strength) || strength <= 0 || strength > 4)
            throw new ArgumentOutOfRangeException(nameof(strength), "Filter strength must be finite, positive, and at most four.");
        if (searchRadius < 1 || searchRadius > 16)
            throw new ArgumentOutOfRangeException(nameof(searchRadius), "Search radius must be from one through sixteen.");
        if (patchRadius < 0 || patchRadius > Border)
            throw new ArgumentOutOfRangeException(nameof(patchRadius), "Patch radius must be from zero through four.");
        hasFiltered = false;
        GpuView<DenoisingColor> input = noisy.DeviceView;
        GpuView<DenoisingColor> result = filtered.DeviceView;
        GpuView<float> totalWeights = weights.DeviceView;
        // A self weight of one also keeps zero-noise and tiny-image cases well defined.
        gpu.For(0, pixelCount, index =>
        {
            result[index] = input[index];
            totalWeights[index] = 1;
        });
        if (sigma > 0)
        {
            float noiseVariance = 2f * sigma * sigma;
            // Double arithmetic on the host avoids overflow/underflow when deriving the width.
            // Clamp its inverse to keep valid, extremely small sigma/strength values well defined.
            double filterWidth = (double)strength * sigma;
            float inverseWidthSquared = (float)Math.Min(1e20, 1.0 / (filterWidth * filterWidth));
            int patchSide = 2 * patchRadius + 1;
            float inversePatchArea = 1f / (patchSide * patchSide);
            for (int dy = -searchRadius; dy <= searchRadius; dy++)
            {
                for (int dx = -searchRadius; dx <= searchRadius; dx++)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    AccumulateOffset(dx, dy, patchRadius, inversePatchArea, noiseVariance, inverseWidthSquared);
                }
            }
            gpu.For(0, pixelCount, index => result[index] = result[index].Scale(1f / totalWeights[index]));
        }
        hasFiltered = true;
    }

    /// <summary>Converts the noisy image to packed RGB bytes on the GPU and downloads it.</summary>
    public byte[] GetNoisyPixels()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        EnsureNoisy();
        return GetPixels(noisy);
    }

    /// <summary>Converts the filtered image to packed RGB bytes on the GPU and downloads it.</summary>
    public byte[] GetDenoisedPixels()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        EnsureFiltered();
        return GetPixels(filtered);
    }

    /// <summary>Computes byte-quantized RGB squared error and its reduction on the GPU.</summary>
    /// <param name="denoised">True to measure the filtered result, false to measure the noisy image.</param>
    /// <returns>PSNR in decibels relative to the uploaded original, or positive infinity for identical pixels.</returns>
    /// <remarks>The final scalar logarithm is evaluated on the host; no per-pixel metric work runs there.</remarks>
    public double GetPsnr(bool denoised)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (denoised)
            EnsureFiltered();
        else
            EnsureNoisy();
        GpuView<byte> reference = original.DeviceView;
        GpuView<DenoisingColor> image = (denoised ? filtered : noisy).DeviceView;
        GpuView<long> errors = squaredErrors.DeviceView;
        gpu.For(0, pixelCount, index =>
        {
            DenoisingColor color = image[index];
            int offset = index * 3;
            int red = Quantize(color.R) - reference[offset];
            int green = Quantize(color.G) - reference[offset + 1];
            int blue = Quantize(color.B) - reference[offset + 2];
            errors[index] = (long)(red * red + green * green + blue * blue);
        });
        long sum = gpu.Reduce(squaredErrors, new Int64Sum());
        return sum == 0 ? double.PositiveInfinity : 10 * Math.Log10(255.0 * 255 * (3.0 * pixelCount) / sum);
    }

    /// <summary>Releases all owned device buffers without disposing the borrowed executor.</summary>
    public void Dispose()
    {
        if (disposed)
            return;
        squaredErrors.Dispose();
        weights.Dispose();
        horizontalSums.Dispose();
        differences.Dispose();
        rgbOutput.Dispose();
        filtered.Dispose();
        noisy.Dispose();
        original.Dispose();
        disposed = true;
    }

    /// <summary>Adds one neighboring search offset using squared differences and separable patch sums.</summary>
    private void AccumulateOffset(int dx, int dy, int patchRadius, float inversePatchArea,
        float noiseVariance, float inverseWidthSquared)
    {
        GpuView<DenoisingColor> input = noisy.DeviceView;
        GpuView<DenoisingColor> result = filtered.DeviceView;
        GpuView<float> difference = differences.DeviceView;
        GpuView<float> horizontal = horizontalSums.DeviceView;
        GpuView<float> totalWeights = weights.DeviceView;
        int imageWidth = width;
        int imageHeight = height;
        int differenceWidth = paddedWidth;
        gpu.For(0, differences.Length, index =>
        {
            int x = index % differenceWidth - Border;
            int y = index / differenceWidth - Border;
            DenoisingColor first = input[Clamp(y, imageHeight) * imageWidth + Clamp(x, imageWidth)];
            DenoisingColor second = input[Clamp(y + dy, imageHeight) * imageWidth + Clamp(x + dx, imageWidth)];
            float red = first.R - second.R;
            float green = first.G - second.G;
            float blue = first.B - second.B;
            difference[index] = (red * red + green * green + blue * blue) * (1f / 3f);
        });
        gpu.For(0, horizontalSums.Length, index =>
        {
            int x = index % imageWidth;
            int row = index / imageWidth;
            int first = row * differenceWidth + x + Border - patchRadius;
            float sum = 0;
            for (int patchX = 0; patchX <= 2 * patchRadius; patchX++)
                sum += difference[first + patchX];
            horizontal[index] = sum;
        });
        gpu.For(0, pixelCount, index =>
        {
            int x = index % imageWidth;
            int y = index / imageWidth;
            int neighborX = x + dx;
            int neighborY = y + dy;
            if (neighborX >= 0 && neighborX < imageWidth && neighborY >= 0 && neighborY < imageHeight)
            {
                int first = (y + Border - patchRadius) * imageWidth + x;
                float sum = 0;
                for (int patchY = 0; patchY <= 2 * patchRadius; patchY++)
                    sum += horizontal[first + patchY * imageWidth];
                float distance = MathF.Max(0, sum * inversePatchArea - noiseVariance);
                float weight = MathF.Exp(-distance * inverseWidthSquared);
                DenoisingColor neighbor = input[neighborY * imageWidth + neighborX];
                result[index] = result[index].Add(neighbor.Scale(weight));
                totalWeights[index] += weight;
            }
        });
    }

    /// <summary>Converts one device image to reusable RGB-byte storage and downloads that storage.</summary>
    private byte[] GetPixels(GpuBuffer<DenoisingColor> image)
    {
        GpuView<DenoisingColor> input = image.DeviceView;
        GpuView<byte> output = rgbOutput.DeviceView;
        gpu.For(0, pixelCount, index =>
        {
            DenoisingColor color = input[index];
            int offset = index * 3;
            output[offset] = Quantize(color.R);
            output[offset + 1] = Quantize(color.G);
            output[offset + 2] = Quantize(color.B);
        });
        return rgbOutput.ToArray();
    }

    /// <summary>Checks that an operation has a prepared noisy image.</summary>
    private void EnsureNoisy()
    {
        if (!hasNoisy)
            throw new InvalidOperationException("Generate noise or select the input image before processing it.");
    }

    /// <summary>Checks that an operation has a current filtered image.</summary>
    private void EnsureFiltered()
    {
        if (!hasFiltered)
            throw new InvalidOperationException("Denoise the current noisy image before downloading or measuring it.");
    }

    /// <summary>Checks the byte-intensity range of a finite noise standard deviation.</summary>
    private static void ValidateSigma(float sigma)
    {
        if (!float.IsFinite(sigma) || sigma < 0 || sigma > 100)
            throw new ArgumentOutOfRangeException(nameof(sigma), "Noise sigma must be finite and from zero through 100.");
    }

    /// <summary>Replicates image-edge coordinates for patch comparisons.</summary>
    private static int Clamp(int coordinate, int length) => Math.Min(Math.Max(coordinate, 0), length - 1);

    /// <summary>Clamps and rounds a channel to the exact byte written to PNG files.</summary>
    private static byte Quantize(float value) => (byte)(MathF.Min(255f, MathF.Max(0, value)) + 0.5f);

    /// <summary>Produces one standard-normal sample by Box-Muller from the seed and absolute RGB index.</summary>
    private static float Gaussian(int channelIndex, ulong seed)
    {
        uint state = MixBits(unchecked((uint)channelIndex + 0x9e3779b9u));
        state = MixBits(state ^ (uint)seed ^ MixBits((uint)(seed >> 32)));
        if (state == 0)
            state = 0x6d2b79f5u;
        float first = NextUniform(ref state);
        float second = NextUniform(ref state);
        return MathF.Sqrt(-2f * MathF.Log(first)) * MathF.Cos(6.283185307179586f * second);
    }

    /// <summary>Hashes a 32-bit value with wrapping integer multiplication.</summary>
    private static uint MixBits(uint value)
    {
        value ^= value >> 16;
        value = unchecked(value * 0x7feb352du);
        value ^= value >> 15;
        value = unchecked(value * 0x846ca68bu);
        return value ^ (value >> 16);
    }

    /// <summary>Advances xorshift32 and returns a uniform value strictly above zero.</summary>
    private static float NextUniform(ref uint state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return ((state >> 8) + 0.5f) * (1f / 16777216f);
    }
}

/// <summary>Stores device-compatible RGB channels in byte-intensity units.</summary>
public readonly struct DenoisingColor
{
    /// <summary>The red channel.</summary>
    public readonly float R;
    /// <summary>The green channel.</summary>
    public readonly float G;
    /// <summary>The blue channel.</summary>
    public readonly float B;

    /// <summary>Creates an RGB color from three channel values.</summary>
    public DenoisingColor(float red, float green, float blue)
    {
        R = red;
        G = green;
        B = blue;
    }

    /// <summary>Adds RGB channels component by component.</summary>
    public DenoisingColor Add(DenoisingColor other) => new(R + other.R, G + other.G, B + other.B);

    /// <summary>Scales all three RGB channels by a common weight.</summary>
    public DenoisingColor Scale(float factor) => new(R * factor, G * factor, B * factor);
}
