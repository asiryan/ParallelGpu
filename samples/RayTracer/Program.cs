using System.Diagnostics;
using ParallelGpu;
using RayTracer;

// Change the rendering parameters here before running the sample.
const int Width = 960;
const int Height = 540;
const int SamplesPerPixel = 128;
const int MaxBounces = 16;
const ulong Seed = 42;
const string OutputFile = "artifacts/raytracer.png";
bool saveCheckpoints = false;

try
{
    string outputPath = Path.GetFullPath(OutputFile);

    // Auto selects CUDA first, then compatible OpenCL devices; a real GPU is required.
    using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Image: {Width} x {Height}; {SamplesPerPixel} samples/pixel; {MaxBounces} bounces; seed {Seed}");
    Console.WriteLine("Scene: glass, polished and rough metal, and diffuse spheres on a checkerboard ground.");

    // Compile the sampling, RGB conversion, and reset kernels on a small image first.
    var setupTimer = Stopwatch.StartNew();
    using (var warmUp = new RayTracingRenderer(gpu, 16, 16, MaxBounces, Seed))
    {
        warmUp.AddSamples(1);
        warmUp.GetRgbPixels();
        warmUp.Reset();
    }
    setupTimer.Stop();
    Console.WriteLine($"Kernel warm-up: {setupTimer.Elapsed.TotalMilliseconds:F3} ms");

    using var renderer = new RayTracingRenderer(gpu, Width, Height, MaxBounces, Seed);
    bool cancelled = false;
    ConsoleCancelEventHandler cancelHandler = (_, e) => { e.Cancel = true; Volatile.Write(ref cancelled, true); };
    Console.CancelKeyPress += cancelHandler;
    try
    {
        int[] checkpoints = saveCheckpoints ? [1, 8, 32] : [];
        var renderTimer = new Stopwatch();
        while (renderer.SamplesPerPixel < SamplesPerPixel && !Volatile.Read(ref cancelled))
        {
            int target = SamplesPerPixel;
            foreach (int checkpoint in checkpoints)
                if (checkpoint > renderer.SamplesPerPixel && checkpoint < target)
                    target = checkpoint;

            int batch = Math.Min(8, target - renderer.SamplesPerPixel);
            renderTimer.Start();
            renderer.AddSamples(batch);
            renderTimer.Stop();

            if (renderer.SamplesPerPixel == target || renderer.SamplesPerPixel % 32 == 0)
                Console.WriteLine($"Rendered: {renderer.SamplesPerPixel}/{SamplesPerPixel} samples/pixel; GPU sampling: {renderTimer.Elapsed.TotalSeconds:F3} s");

            if (renderer.SamplesPerPixel == target && target < SamplesPerPixel)
            {
                string checkpointPath = Path.Combine(Path.GetDirectoryName(outputPath)!,
                    $"{Path.GetFileNameWithoutExtension(outputPath)}-{target:D3}spp.png");
                RayTracerImage.SavePng(checkpointPath, renderer.GetRgbPixels(), Width, Height);
                Console.WriteLine($"Saved PNG: {checkpointPath}");
            }
        }

        if (renderer.SamplesPerPixel == 0)
        {
            Console.Error.WriteLine("Rendering cancelled before any samples completed.");
            return 130;
        }

        var outputTimer = Stopwatch.StartNew();
        RayTracerImage.SavePng(outputPath, renderer.GetRgbPixels(), Width, Height);
        outputTimer.Stop();
        Console.WriteLine($"GPU RGB conversion, download, and PNG encoding: {outputTimer.Elapsed.TotalMilliseconds:F3} ms");
        Console.WriteLine($"Saved PNG: {outputPath} ({renderer.SamplesPerPixel} samples/pixel)");
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
