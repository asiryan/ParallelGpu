using ParallelGpu;

namespace RayTracer;

/// <summary>Accumulates stochastic ray-traced samples and converts them to RGB entirely on the GPU.</summary>
/// <remarks>Only scene construction, buffer transfers, and PNG encoding run on the host.
/// Reuse this renderer to retain both compiled kernels and the accumulated device image.</remarks>
public sealed class RayTracingRenderer : IDisposable
{
    /// <summary>The executor used for all rendering and image conversion.</summary>
    private readonly Gpu gpu;
    /// <summary>The image width in pixels.</summary>
    private readonly int width;
    /// <summary>The image height in pixels.</summary>
    private readonly int height;
    /// <summary>The maximum number of surface interactions per ray.</summary>
    private readonly int maxBounces;
    /// <summary>The user seed mixed with each pixel and absolute sample index.</summary>
    private readonly ulong seed;
    /// <summary>The immutable perspective camera description.</summary>
    private readonly RayTracingCamera camera;
    /// <summary>The scene geometry uploaded once and retained on the GPU.</summary>
    private readonly GpuBuffer<RayTracingSphere> spheres;
    /// <summary>The accumulated linear RGB radiance for each pixel.</summary>
    private readonly GpuBuffer<Float3> accumulatedColors;
    /// <summary>The RGB byte image produced by the GPU before downloading.</summary>
    private readonly GpuBuffer<byte> rgbPixels;
    /// <summary>Whether the owned device buffers have been released.</summary>
    private bool disposed;

    /// <summary>Creates a GPU-only renderer and uploads its scene.</summary>
    /// <param name="gpu">A CUDA or OpenCL executor; the CPU backend is rejected.</param>
    /// <param name="width">The positive image width.</param>
    /// <param name="height">The positive image height.</param>
    /// <param name="maxBounces">The positive limit on surface interactions along each ray.</param>
    /// <param name="seed">The deterministic seed for pixel sampling and material scattering.</param>
    public RayTracingRenderer(Gpu gpu, int width, int height, int maxBounces, ulong seed)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        if (!gpu.IsGpu)
            throw new ArgumentException("RayTracer requires a GPU; CPU rendering is not supported.", nameof(gpu));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBounces);
        int pixelCount = checked(width * height);
        int rgbLength = checked(pixelCount * 3);
        this.gpu = gpu;
        this.width = width;
        this.height = height;
        this.maxBounces = maxBounces;
        this.seed = seed;
        camera = RayTracingScene.CreateCamera(width, height);
        spheres = gpu.Upload(RayTracingScene.CreateSpheres());
        try
        {
            accumulatedColors = gpu.Allocate<Float3>(pixelCount);
            try
            {
                rgbPixels = gpu.Allocate<byte>(rgbLength);
            }
            catch
            {
                accumulatedColors.Dispose();
                throw;
            }
        }
        catch
        {
            spheres.Dispose();
            throw;
        }
    }

    /// <summary>Gets the number of samples currently accumulated for every pixel.</summary>
    public int SamplesPerPixel { get; private set; }

    /// <summary>Adds deterministic samples to the resident image using at most eight samples per launch.</summary>
    /// <param name="sampleCount">The positive number of additional samples for each pixel.</param>
    /// <remarks>The absolute sample index determines the random sequence, so splitting work into
    /// different batches produces the same accumulation on a given backend.</remarks>
    public void AddSamples(int sampleCount)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleCount);
        int targetSampleCount = checked(SamplesPerPixel + sampleCount);
        GpuView<Float3> colors = accumulatedColors.DeviceView;
        GpuView<RayTracingSphere> scene = spheres.DeviceView;
        RayTracingCamera sampleCamera = camera;
        int imageWidth = width;
        int imageHeight = height;
        int bounceLimit = maxBounces;
        ulong randomSeed = seed;
        while (SamplesPerPixel < targetSampleCount)
        {
            int firstSample = SamplesPerPixel;
            int batchSize = Math.Min(8, targetSampleCount - firstSample);
            gpu.For(0, accumulatedColors.Length, index =>
                AccumulatePixel(index, colors, scene, sampleCamera, imageWidth, imageHeight,
                    bounceLimit, randomSeed, firstSample, batchSize));
            SamplesPerPixel += batchSize;
        }
    }

    /// <summary>Converts the accumulated image to top-to-bottom RGB bytes on the GPU and downloads it.</summary>
    /// <returns>Three bytes per pixel, ordered red, green, blue.</returns>
    public byte[] GetRgbPixels()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (SamplesPerPixel == 0)
            throw new InvalidOperationException("Accumulate at least one sample before downloading an image.");
        GpuView<Float3> colors = accumulatedColors.DeviceView;
        GpuView<byte> output = rgbPixels.DeviceView;
        float sampleScale = 1f / SamplesPerPixel;
        gpu.For(0, accumulatedColors.Length, index =>
        {
            Float3 color = colors[index] * sampleScale;
            int offset = index * 3;
            output[offset] = ToDisplayByte(color.X);
            output[offset + 1] = ToDisplayByte(color.Y);
            output[offset + 2] = ToDisplayByte(color.Z);
        });
        return rgbPixels.ToArray();
    }

    /// <summary>Clears the accumulated image on the GPU and restarts the deterministic sample sequence.</summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        GpuView<Float3> colors = accumulatedColors.DeviceView;
        gpu.For(0, accumulatedColors.Length, index => colors[index] = default);
        SamplesPerPixel = 0;
    }

    /// <summary>Releases scene, accumulation, and RGB device buffers without disposing their executor.</summary>
    public void Dispose()
    {
        if (disposed)
            return;
        rgbPixels.Dispose();
        accumulatedColors.Dispose();
        spheres.Dispose();
        disposed = true;
    }

    /// <summary>Traces independent jittered rays and adds them in sample order to one pixel.</summary>
    private static void AccumulatePixel(int index, GpuView<Float3> colors, GpuView<RayTracingSphere> scene,
        RayTracingCamera camera, int width, int height, int maxBounces, ulong seed, int firstSample, int count)
    {
        Float3 color = colors[index];
        int x = index % width;
        int y = index / width;
        for (int sample = firstSample; sample < firstSample + count; sample++)
        {
            uint random = CreateRandomState(index, sample, seed);
            float u = (x + NextRandom(ref random)) / width;
            float v = (height - y - NextRandom(ref random)) / height;
            Float3 direction = Float3.Normalize(camera.LowerLeft + camera.Horizontal * u +
                camera.Vertical * v - camera.Origin);
            color += TraceRay(camera.Origin, direction, scene, maxBounces, ref random);
        }
        colors[index] = color;
    }

    /// <summary>Follows a ray through diffuse, metallic, and dielectric interactions until it reaches the sky.</summary>
    private static Float3 TraceRay(Float3 origin, Float3 direction, GpuView<RayTracingSphere> scene,
        int maxBounces, ref uint random)
    {
        var throughput = new Float3(1, 1, 1);
        for (int bounce = 0; bounce < maxBounces; bounce++)
        {
            int closestSphere = -1;
            float closestDistance = float.MaxValue;
            for (int sphereIndex = 0; sphereIndex < scene.Length; sphereIndex++)
            {
                RayTracingSphere sphere = scene[sphereIndex];
                Float3 offset = origin - sphere.Center;
                float a = Float3.Dot(direction, direction);
                float halfB = Float3.Dot(offset, direction);
                float c = Float3.Dot(offset, offset) - sphere.Radius * sphere.Radius;
                float discriminant = halfB * halfB - a * c;
                if (discriminant < 0)
                    continue;
                float squareRoot = MathF.Sqrt(discriminant);
                float distance = (-halfB - squareRoot) / a;
                if (distance <= 0.001f)
                    distance = (-halfB + squareRoot) / a;
                if (distance > 0.001f && distance < closestDistance)
                {
                    closestDistance = distance;
                    closestSphere = sphereIndex;
                }
            }
            if (closestSphere < 0)
                return throughput * SkyColor(direction);

            RayTracingSphere hitSphere = scene[closestSphere];
            Float3 hitPoint = origin + direction * closestDistance;
            Float3 outwardNormal = (hitPoint - hitSphere.Center) * (1f / hitSphere.Radius);
            bool frontFace = Float3.Dot(direction, outwardNormal) < 0;
            Float3 normal = frontFace ? outwardNormal : -outwardNormal;
            if (hitSphere.Material == 2)
            {
                float ratio = frontFace ? 1f / hitSphere.RefractionIndex : hitSphere.RefractionIndex;
                float cosTheta = MathF.Min(Float3.Dot(-direction, normal), 1f);
                float sinSquared = MathF.Max(0, 1f - cosTheta * cosTheta);
                if (ratio * ratio * sinSquared > 1f || Reflectance(cosTheta, ratio) > NextRandom(ref random))
                    direction = Reflect(direction, normal);
                else
                {
                    Float3 perpendicular = (direction + normal * cosTheta) * ratio;
                    Float3 parallel = normal * -MathF.Sqrt(MathF.Max(0, 1f - Float3.Dot(perpendicular, perpendicular)));
                    direction = perpendicular + parallel;
                }
            }
            else if (hitSphere.Material == 1)
            {
                direction = Float3.Normalize(Reflect(direction, normal) + RandomInUnitSphere(ref random) * hitSphere.Roughness);
                if (Float3.Dot(direction, normal) <= 0)
                    return default;
                throughput *= hitSphere.Albedo;
            }
            else
            {
                direction = Float3.Normalize(normal + Float3.Normalize(RandomInUnitSphere(ref random)));
                if (Float3.Dot(direction, direction) < 0.5f)
                    direction = normal;
                Float3 albedo = hitSphere.Albedo;
                if (hitSphere.Material == 3)
                {
                    int checker = (int)MathF.Floor(hitPoint.X * 1.2f) + (int)MathF.Floor(hitPoint.Z * 1.2f);
                    albedo = (checker & 1) == 0 ? new(0.72f, 0.69f, 0.62f) : new(0.2f, 0.25f, 0.28f);
                }
                throughput *= albedo;
            }
            direction = Float3.Normalize(direction);
            // Offset toward the outgoing side, including the inside of a refracting sphere.
            float offsetSign = Float3.Dot(direction, outwardNormal) >= 0 ? 0.001f : -0.001f;
            origin = hitPoint + outwardNormal * offsetSign;
        }
        // The explicit bounce limit truncates paths that have not reached the environment.
        return default;
    }

    /// <summary>Samples a bright sky gradient that provides the scene's environmental illumination.</summary>
    private static Float3 SkyColor(Float3 direction)
    {
        float blend = 0.5f * (direction.Y + 1f);
        return new Float3(0.95f, 0.92f, 0.85f) * (1f - blend) + new Float3(0.45f, 0.65f, 1f) * blend;
    }

    /// <summary>Reflects an incident unit vector about an oriented surface normal.</summary>
    private static Float3 Reflect(Float3 direction, Float3 normal) =>
        direction - normal * (2f * Float3.Dot(direction, normal));

    /// <summary>Computes Schlick's approximation of Fresnel reflection probability for glass.</summary>
    private static float Reflectance(float cosine, float ratio)
    {
        float baseReflectance = (1f - ratio) / (1f + ratio);
        baseReflectance *= baseReflectance;
        float complement = 1f - cosine;
        float square = complement * complement;
        return baseReflectance + (1f - baseReflectance) * square * square * complement;
    }

    /// <summary>Samples the unit ball with a fixed rejection limit to bound per-thread work.</summary>
    private static Float3 RandomInUnitSphere(ref uint random)
    {
        for (int attempt = 0; attempt < 16; attempt++)
        {
            var candidate = new Float3(2f * NextRandom(ref random) - 1f,
                2f * NextRandom(ref random) - 1f, 2f * NextRandom(ref random) - 1f);
            float squaredLength = Float3.Dot(candidate, candidate);
            if (squaredLength > 1e-12f && squaredLength < 1f)
                return candidate;
        }
        // A rare fallback avoids an unbounded rejection loop without introducing a zero vector.
        return new(0.5f, 0.5f, 0.5f);
    }

    /// <summary>Mixes both seed halves with the pixel and absolute sample index into a nonzero RNG state.</summary>
    private static uint CreateRandomState(int index, int sample, ulong seed)
    {
        uint state = MixBits(unchecked((uint)index + 0x9e3779b9u));
        state = MixBits(state ^ MixBits(unchecked((uint)sample + 0x85ebca6bu)));
        state = MixBits(state ^ (uint)seed ^ MixBits((uint)(seed >> 32)));
        return state == 0 ? 0x6d2b79f5u : state;
    }

    /// <summary>Hashes a 32-bit value using shifts and wrapping integer multiplication.</summary>
    private static uint MixBits(uint value)
    {
        value ^= value >> 16;
        value = unchecked(value * 0x7feb352du);
        value ^= value >> 15;
        value = unchecked(value * 0x846ca68bu);
        return value ^ (value >> 16);
    }

    /// <summary>Advances xorshift32 and returns a uniformly spaced value in [0, 1).</summary>
    private static float NextRandom(ref uint state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (state >> 8) * (1f / 16777216f);
    }

    /// <summary>Applies fixed exposure, Reinhard tone mapping, and square-root gamma encoding on the GPU.</summary>
    private static byte ToDisplayByte(float linear)
    {
        float exposed = MathF.Max(0, linear) * 1.5f;
        float mapped = exposed / (1f + exposed);
        float encoded = MathF.Sqrt(mapped);
        return (byte)(MathF.Min(1, encoded) * 255f + 0.5f);
    }
}
