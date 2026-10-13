using System.Diagnostics;
using ParallelGpu;
using RayTracer;

try
{
    RenderSettings settings = RenderSettings.Parse(args);
    if (settings.ShowHelp)
    {
        Console.WriteLine("Usage: RayTracer [output.png] [--width N] [--height N] [--samples N] [--bounces N] [--seed N] [--checkpoints]");
        Console.WriteLine("Defaults: artifacts/raytracer.png, 960 x 540, 128 samples/pixel, 16 bounces, seed 42.");
        Console.WriteLine("--checkpoints also saves images at 1, 8, and 32 samples/pixel when below the final sample count.");
        return 0;
    }

    // Auto selects CUDA first, then compatible OpenCL devices; a real GPU is required.
    using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Image: {settings.Width} x {settings.Height}; {settings.SamplesPerPixel} samples/pixel; {settings.MaxBounces} bounces; seed {settings.Seed}");
    Console.WriteLine("Scene: glass, polished and rough metal, and diffuse spheres on a checkerboard ground.");

    // Compile the sampling, RGB conversion, and reset kernels on a small image first.
    var setupTimer = Stopwatch.StartNew();
    using (var warmUp = new RayTracingRenderer(gpu, 16, 16, settings.MaxBounces, settings.Seed))
    {
        warmUp.AddSamples(1);
        warmUp.GetRgbPixels();
        warmUp.Reset();
    }
    setupTimer.Stop();
    Console.WriteLine($"Kernel warm-up: {setupTimer.Elapsed.TotalMilliseconds:F3} ms");

    using var renderer = new RayTracingRenderer(gpu, settings.Width, settings.Height, settings.MaxBounces, settings.Seed);
    bool cancelled = false;
    ConsoleCancelEventHandler cancelHandler = (_, e) => { e.Cancel = true; Volatile.Write(ref cancelled, true); };
    Console.CancelKeyPress += cancelHandler;
    try
    {
        int[] checkpoints = settings.SaveCheckpoints ? [1, 8, 32] : [];
        var renderTimer = new Stopwatch();
        while (renderer.SamplesPerPixel < settings.SamplesPerPixel && !Volatile.Read(ref cancelled))
        {
            int target = settings.SamplesPerPixel;
            foreach (int checkpoint in checkpoints)
                if (checkpoint > renderer.SamplesPerPixel && checkpoint < target)
                    target = checkpoint;

            int batch = Math.Min(8, target - renderer.SamplesPerPixel);
            renderTimer.Start();
            renderer.AddSamples(batch);
            renderTimer.Stop();

            if (renderer.SamplesPerPixel == target || renderer.SamplesPerPixel % 32 == 0)
                Console.WriteLine($"Rendered: {renderer.SamplesPerPixel}/{settings.SamplesPerPixel} samples/pixel; GPU sampling: {renderTimer.Elapsed.TotalSeconds:F3} s");

            if (renderer.SamplesPerPixel == target && target < settings.SamplesPerPixel)
            {
                string checkpointPath = Path.Combine(Path.GetDirectoryName(settings.OutputPath)!,
                    $"{Path.GetFileNameWithoutExtension(settings.OutputPath)}-{target:D3}spp.png");
                RayTracerImage.SavePng(checkpointPath, renderer.GetRgbPixels(), settings.Width, settings.Height);
                Console.WriteLine($"Saved PNG: {checkpointPath}");
            }
        }

        if (renderer.SamplesPerPixel == 0)
        {
            Console.Error.WriteLine("Rendering cancelled before any samples completed.");
            return 130;
        }

        var outputTimer = Stopwatch.StartNew();
        RayTracerImage.SavePng(settings.OutputPath, renderer.GetRgbPixels(), settings.Width, settings.Height);
        outputTimer.Stop();
        Console.WriteLine($"GPU RGB conversion, download, and PNG encoding: {outputTimer.Elapsed.TotalMilliseconds:F3} ms");
        Console.WriteLine($"Saved PNG: {settings.OutputPath} ({renderer.SamplesPerPixel} samples/pixel)");
        return Volatile.Read(ref cancelled) ? 130 : 0;
    }
    finally
    {
        Console.CancelKeyPress -= cancelHandler;
    }
}
catch (Exception error) when (error is ArgumentException or InvalidOperationException or OverflowException
    or IOException or UnauthorizedAccessException or NotSupportedException)
{
    Console.Error.WriteLine($"RayTracer failed: {error.Message}");
    return 1;
}
