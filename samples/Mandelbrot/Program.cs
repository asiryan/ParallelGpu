using System.Diagnostics;
using Mandelbrot;
using ParallelGpu;

const int Width = 1600;
const int Height = 1000;
const int MaxIterations = 1000;

if (args.Length > 1)
{
    Console.Error.WriteLine("Usage: Mandelbrot [output.png]");
    return 1;
}

try
{
    string outputPath = Path.GetFullPath(args.Length == 1 ? args[0] : Path.Combine("artifacts", "mandelbrot.png"));
    using var gpu = new Gpu(new GpuOptions
    {
        Backend = GpuBackend.Auto,
        AllowCpuFallback = true
    });

    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Image: {Width} x {Height}; iteration limit: {MaxIterations}");
    Console.WriteLine("Operation: Mandelbrot escape counts, z = z² + c, one pixel per iteration");

    // Warm up CPU JIT and the reusable GPU lambda before measuring the full image.
    MandelbrotRenderer.RenderOnCpu(32, 20, 32);
    MandelbrotRenderer.RenderOnGpu(gpu, 32, 20, 32);

    var timer = Stopwatch.StartNew();
    int[] cpuIterations = MandelbrotRenderer.RenderOnCpu(Width, Height, MaxIterations);
    timer.Stop();
    Console.WriteLine($"CPU Parallel.For: {timer.Elapsed.TotalMilliseconds:F3} ms");

    timer.Restart();
    int[] gpuIterations = MandelbrotRenderer.RenderOnGpu(gpu, Width, Height, MaxIterations);
    timer.Stop();
    Console.WriteLine($"{gpu.Backend} For<int>, including allocations and array transfers: {timer.Elapsed.TotalMilliseconds:F3} ms");

    int differentPixels = 0;
    for (int i = 0; i < cpuIterations.Length; i++)
        if (cpuIterations[i] != gpuIterations[i])
            differentPixels++;
    Console.WriteLine($"CPU/device escape-count differences: {differentPixels:N0} pixels");
    // Floating-point rounding may change escape counts near the fractal boundary.

    // Coloring and PNG encoding run on the host, outside the compute timings.
    MandelbrotImage.SavePng(outputPath, gpuIterations, Width, Height, MaxIterations);
    Console.WriteLine($"Saved PNG: {outputPath}");
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidOperationException or OverflowException
    or IOException or UnauthorizedAccessException or NotSupportedException)
{
    Console.Error.WriteLine($"Mandelbrot failed: {error.Message}");
    return 1;
}
