using System.Diagnostics;
using MonteCarloPi;
using ParallelGpu;

// Change the number of points and seed here. The iteration size controls GPU work distribution.
const long SampleCount = 100_000_000;
const ulong Seed = 12345;
const int SamplesPerIteration = 256;

try
{
    var totalTimer = Stopwatch.StartNew();
    // Auto selects CUDA or compatible OpenCL; CPU fallback is disabled.
    using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
    var estimator = new PiEstimator(gpu,
        sampleCount: SampleCount,
        seed: Seed,
        samplesPerIteration: SamplesPerIteration);

    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Samples: {SampleCount:N0}; seed: {Seed}; samples per iteration: {SamplesPerIteration}");

    // Exclude device setup and initial kernel compilation from the calculation timing.
    estimator.WarmUp();
    Console.WriteLine($"Initialization and kernel warm-up: {totalTimer.Elapsed.TotalMilliseconds:F3} ms");

    var timer = Stopwatch.StartNew();
    PiEstimate result = estimator.Estimate();
    timer.Stop();
    totalTimer.Stop();

    Console.WriteLine($"Inside quarter-circle: {result.InsideCircle:N0}");
    Console.WriteLine($"Estimated pi: {result.Value:F12}");
    Console.WriteLine($"Math.PI:      {Math.PI:F12}");
    // Monte Carlo is approximate: show the observed error without imposing a fixed tolerance.
    Console.WriteLine($"Absolute error: {Math.Abs(result.Value - Math.PI):E6}");
    Console.WriteLine($"GPU sampling and reduction, including allocation and result transfer: {timer.Elapsed.TotalMilliseconds:F3} ms");
    Console.WriteLine($"Throughput: {SampleCount / timer.Elapsed.TotalSeconds:N0} samples/s");
    Console.WriteLine($"Total including initialization: {totalTimer.Elapsed.TotalMilliseconds:F3} ms");
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidOperationException)
{
    Console.Error.WriteLine($"MonteCarloPi failed: {error.Message}");
    return 1;
}
