using System.Diagnostics;
using MatrixMultiplication;
using ParallelGpu;

// Configure the sample here.
const int Rows = 2048;
const int Inner = 2048;
const int Columns = 2048;
const GpuBackend Backend = GpuBackend.Auto;

float[][] a = Matrices.Random(Rows, Inner, seed: 12345);
float[][] b = Matrices.Random(Inner, Columns, seed: 67890);

using var gpu = new Gpu(new GpuOptions
{
    Backend = Backend,
    AllowCpuFallback = true
});

Console.WriteLine($"A: {Rows} x {Inner}, B: {Inner} x {Columns}");

// Warm up both methods; exclude the initial GPU kernel compilation from timing.
MatrixProduct.MultiplyOnCpu(a, b);
MatrixProduct.MultiplyOnGpu(gpu, a, b);

// CPU
var timer = Stopwatch.StartNew();
float[][] cpuResult = MatrixProduct.MultiplyOnCpu(a, b);
timer.Stop();
Console.WriteLine($"CPU Parallel.For: {timer.Elapsed.TotalMilliseconds:F3} ms");

// GPU
Console.WriteLine($"Device GPU: {gpu.DeviceName} ({gpu.Backend})");

timer.Restart();
float[][] gpuResult = MatrixProduct.MultiplyOnGpu(gpu, a, b);
timer.Stop();
Console.WriteLine($"{gpu.Backend} For<float>, including array transfers: {timer.Elapsed.TotalMilliseconds:F3} ms");

double maxError = Matrices.CheckEqual(cpuResult, gpuResult);
Console.WriteLine($"Result: {Rows} x {Columns}. Validation passed, maximum error: {maxError:G6}");

