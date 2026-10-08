using System.Diagnostics;
using System.Numerics;
using ComplexVectorMultiplication;
using ParallelGpu;

// Configure the sample here.
const int Length = 2 << 24;

Complex[] a = Vectors.Random(Length, seed: 12345);
Complex[] b = Vectors.Random(Length, seed: 67890);

using var gpu = new Gpu(new GpuOptions
{
    Backend = GpuBackend.Auto,
    AllowCpuFallback = true
});

Console.WriteLine($"A: {Length} complex values, B: {Length} complex values");
Console.WriteLine("Operation: sum(a[i] * b[i])");

// Warm up both methods; exclude the initial GPU kernel compilation from timing.
VectorProduct.MultiplyOnCpu(a, b);
VectorProduct.MultiplyOnGpu(gpu, a, b);

// CPU
var timer = Stopwatch.StartNew();
Complex cpuResult = VectorProduct.MultiplyOnCpu(a, b);
timer.Stop();
Console.WriteLine($"CPU Parallel.For: {timer.Elapsed.TotalMilliseconds:F3} ms");

// GPU
Console.WriteLine($"Device GPU: {gpu.DeviceName} ({gpu.Backend})");

timer.Restart();
Complex gpuResult = VectorProduct.MultiplyOnGpu(gpu, a, b);
timer.Stop();
Console.WriteLine($"{gpu.Backend} ForEach<Complex>, including reduction and array transfers: {timer.Elapsed.TotalMilliseconds:F3} ms");

double maxError = Vectors.CheckEqual(cpuResult, gpuResult);
Console.WriteLine($"Result: {gpuResult}. Validation passed, maximum error: {maxError:G6}");
