using System.Diagnostics;
using System.Numerics;
using ComplexVectorMultiplication;
using ParallelGpu;

// Each vector occupies 512 MiB: 33,554,432 values of two doubles each.
const int Length = 2 << 24;

try
{
    Complex[] a = Vectors.Random(Length, seed: 12345);
    Complex[] b = Vectors.Random(Length, seed: 67890);

    using var gpu = new Gpu(new GpuOptions
    {
        Backend = GpuBackend.Auto,
        AllowCpuFallback = true
    });

    Console.WriteLine($"A: {Length} complex values, B: {Length} complex values");
    Console.WriteLine("Operation: sum(a[i] * b[i])");

    // Exclude initial CPU JIT and accelerator kernel compilation from timing.
    VectorProduct.MultiplyOnCpu(a, b);
    VectorProduct.MultiplyOnGpu(gpu, a, b);

    var timer = Stopwatch.StartNew();
    Complex cpuResult = VectorProduct.MultiplyOnCpu(a, b);
    timer.Stop();
    Console.WriteLine($"CPU Parallel.For: {timer.Elapsed.TotalMilliseconds:F3} ms");

    Console.WriteLine($"Device GPU: {gpu.DeviceName} ({gpu.Backend})");

    timer.Restart();
    Complex gpuResult = VectorProduct.MultiplyOnGpu(gpu, a, b);
    timer.Stop();
    Console.WriteLine($"{gpu.Backend} For<Complex>, including reduction and array transfers: {timer.Elapsed.TotalMilliseconds:F3} ms");

    double maxError = Vectors.CheckEqual(cpuResult, gpuResult);
    Console.WriteLine($"Result: {gpuResult}. Validation passed, maximum error: {maxError:G6}");
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidOperationException or OverflowException)
{
    Console.Error.WriteLine($"ComplexVectorMultiplication failed: {error.Message}");
    return 1;
}
