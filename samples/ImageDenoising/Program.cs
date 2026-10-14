using System.Diagnostics;
using ImageDenoising;
using ParallelGpu;

try
{
    DenoisingSettings settings = DenoisingSettings.Parse(args);
    if (settings.ShowHelp)
    {
        Console.WriteLine("Usage: ImageDenoising [input.png] [--output DIR] [--sigma N] [--strength N] [--search-radius N] [--patch-radius N] [--seed N] [--skip-noise]");
        Console.WriteLine("Defaults: bundled image, artifacts/image-denoising, sigma 20, strength 0.55, search radius 10, patch radius 1, seed 42.");
        Console.WriteLine("--skip-noise filters an already noisy image, including a RayTracer PNG.");
        Console.WriteLine("Output: original.png, before.png (filter input), after.png (filtered result).");
        return 0;
    }

    RgbImage image = ImageFiles.Load(settings.InputPath);
    using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Input: {settings.InputPath}; {image.Width} x {image.Height}");
    Console.WriteLine(settings.SkipNoise ? "Noise generation: skipped; input is already noisy" : $"Gaussian noise: sigma {settings.Sigma:G}, seed {settings.Seed}; clamped and rounded to RGB bytes");
    Console.WriteLine($"Non-Local Means: {settings.SearchRadius * 2 + 1} x {settings.SearchRadius * 2 + 1} search window; {settings.PatchRadius * 2 + 1} x {settings.PatchRadius * 2 + 1} RGB patches; strength {settings.Strength:G}");

    // Warm up all processing kernels on a tiny image before measuring the actual input.
    var timer = Stopwatch.StartNew();
    using (var warmUp = new GpuImageDenoiser(gpu, new byte[16 * 16 * 3], 16, 16))
    {
        warmUp.AddGaussianNoise(settings.Sigma, settings.Seed);
        warmUp.Denoise(settings.Sigma, settings.Strength, settings.SearchRadius, settings.PatchRadius);
        warmUp.GetNoisyPixels();
        warmUp.GetDenoisedPixels();
        if (!settings.SkipNoise)
        {
            warmUp.GetPsnr(false);
            warmUp.GetPsnr(true);
        }
    }
    timer.Stop();
    Console.WriteLine($"Kernel warm-up: {timer.Elapsed.TotalMilliseconds:F3} ms");

    using var denoiser = new GpuImageDenoiser(gpu, image.Pixels, image.Width, image.Height);
    timer.Restart();
    if (settings.SkipNoise)
        denoiser.UseInputAsNoisy();
    else
        denoiser.AddGaussianNoise(settings.Sigma, settings.Seed);
    timer.Stop();
    Console.WriteLine($"GPU input preparation: {timer.Elapsed.TotalMilliseconds:F3} ms");

    // Save the exact quantized image consumed by the filter before starting denoising.
    string originalPath = Path.Combine(settings.OutputDirectory, "original.png");
    string beforePath = Path.Combine(settings.OutputDirectory, "before.png");
    string afterPath = Path.Combine(settings.OutputDirectory, "after.png");
    ImageFiles.SavePng(originalPath, image.Pixels, image.Width, image.Height);
    ImageFiles.SavePng(beforePath, denoiser.GetNoisyPixels(), image.Width, image.Height);

    timer.Restart();
    denoiser.Denoise(settings.Sigma, settings.Strength, settings.SearchRadius, settings.PatchRadius);
    timer.Stop();
    Console.WriteLine($"GPU denoising: {timer.Elapsed.TotalMilliseconds:F3} ms");
    ImageFiles.SavePng(afterPath, denoiser.GetDenoisedPixels(), image.Width, image.Height);

    if (!settings.SkipNoise)
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
