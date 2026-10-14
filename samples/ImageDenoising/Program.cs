using System.Diagnostics;
using ImageDenoising;
using ParallelGpu;

// Change the input, noise, and filter parameters here before running the sample.
string inputPath = Path.Combine(AppContext.BaseDirectory, "Assets", "input.png");
const string OutputDirectory = "artifacts/image-denoising";
const float Sigma = 20;
const float Strength = 0.55f;
const int SearchRadius = 10;
const int PatchRadius = 1;
const ulong Seed = 42;
bool generateNoise = true;

try
{
    inputPath = Path.GetFullPath(inputPath);
    string outputDirectory = Path.GetFullPath(OutputDirectory);
    foreach (string outputName in new[] { "original.png", "before.png", "after.png" })
        if (string.Equals(inputPath, Path.Combine(outputDirectory, outputName), StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Choose an output directory that does not overwrite the input image.");

    RgbImage image = ImageFiles.Load(inputPath);
    using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Input: {inputPath}; {image.Width} x {image.Height}");
    Console.WriteLine(generateNoise ? $"Gaussian noise: sigma {Sigma:G}, seed {Seed}; clamped and rounded to RGB bytes" : "Noise generation: skipped; input is already noisy");
    Console.WriteLine($"Non-Local Means: {SearchRadius * 2 + 1} x {SearchRadius * 2 + 1} search window; {PatchRadius * 2 + 1} x {PatchRadius * 2 + 1} RGB patches; strength {Strength:G}");

    // Warm up all processing kernels on a tiny image before measuring the actual input.
    var timer = Stopwatch.StartNew();
    using (var warmUp = new GpuImageDenoiser(gpu, new byte[16 * 16 * 3], 16, 16))
    {
        warmUp.AddGaussianNoise(Sigma, Seed);
        warmUp.Denoise(Sigma, Strength, SearchRadius, PatchRadius);
        warmUp.GetNoisyPixels();
        warmUp.GetDenoisedPixels();
        if (generateNoise)
        {
            warmUp.GetPsnr(false);
            warmUp.GetPsnr(true);
        }
    }
    timer.Stop();
    Console.WriteLine($"Kernel warm-up: {timer.Elapsed.TotalMilliseconds:F3} ms");

    using var denoiser = new GpuImageDenoiser(gpu, image.Pixels, image.Width, image.Height);
    timer.Restart();
    if (generateNoise)
        denoiser.AddGaussianNoise(Sigma, Seed);
    else
        denoiser.UseInputAsNoisy();
    timer.Stop();
    Console.WriteLine($"GPU input preparation: {timer.Elapsed.TotalMilliseconds:F3} ms");

    // Save the exact quantized image consumed by the filter before starting denoising.
    string originalPath = Path.Combine(outputDirectory, "original.png");
    string beforePath = Path.Combine(outputDirectory, "before.png");
    string afterPath = Path.Combine(outputDirectory, "after.png");
    ImageFiles.SavePng(originalPath, image.Pixels, image.Width, image.Height);
    ImageFiles.SavePng(beforePath, denoiser.GetNoisyPixels(), image.Width, image.Height);

    timer.Restart();
    denoiser.Denoise(Sigma, Strength, SearchRadius, PatchRadius);
    timer.Stop();
    Console.WriteLine($"GPU denoising: {timer.Elapsed.TotalMilliseconds:F3} ms");
    ImageFiles.SavePng(afterPath, denoiser.GetDenoisedPixels(), image.Width, image.Height);

    if (generateNoise)
    {
        double beforePsnr = denoiser.GetPsnr(false);
        double afterPsnr = denoiser.GetPsnr(true);
        Console.WriteLine($"PSNR against the original: before {beforePsnr:F2} dB; after {afterPsnr:F2} dB (higher is better)");
    }
    Console.WriteLine($"Saved original: {originalPath}");
    Console.WriteLine($"Saved before: {beforePath}");
    Console.WriteLine($"Saved after: {afterPath}");
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidOperationException or OverflowException
    or IOException or UnauthorizedAccessException or NotSupportedException)
{
    Console.Error.WriteLine($"ImageDenoising failed: {error.Message}");
    return 1;
}
