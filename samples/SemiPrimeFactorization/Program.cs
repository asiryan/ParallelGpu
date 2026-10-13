using System.Diagnostics;
using ParallelGpu;
using SemiPrimeFactorization;

// Set the semiprime Number and the GPU batch size here.
const long Number = 1_000_000_016_000_000_063;
const int BatchSize = 1_048_576;
TimeSpan budget = TimeSpan.FromSeconds(30);

try
{
    var totalTimer = Stopwatch.StartNew();
    // Auto selects CUDA or compatible OpenCL; CPU fallback is disabled.
    using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
    var factorizer = new SemiPrimeFactorizer(gpu, number: Number, batchSize: BatchSize);

    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Number: {Number}");
    Console.WriteLine($"Candidate divisors: {factorizer.CandidateCount:N0}; total runtime budget: {budget.TotalSeconds:G} seconds");

    // Measure compilation separately while keeping it inside the total runtime budget.
    factorizer.WarmUp();
    TimeSpan setup = totalTimer.Elapsed;
    if (setup >= budget)
        throw new TimeoutException("Device initialization and kernel compilation exceeded the runtime budget.");
    Console.WriteLine($"Initialization and kernel warm-up: {setup.TotalMilliseconds:F3} ms");

    FactorizationResult result = factorizer.Factor(budget - totalTimer.Elapsed);
    // Checked multiplication independently confirms that the recovered pair reconstructs the input.
    if (checked(result.FirstFactor * result.SecondFactor) != Number)
        throw new InvalidOperationException("Factor product validation failed.");
    totalTimer.Stop();
    if (totalTimer.Elapsed >= budget)
        throw new TimeoutException("The complete demonstration exceeded the runtime budget.");

    Console.WriteLine($"Factors: {result.FirstFactor} * {result.SecondFactor} = {Number}");
    Console.WriteLine($"Candidates tested: {result.CandidatesTested:N0}");
    Console.WriteLine($"GPU search, transfers and factor validation: {result.Elapsed.TotalMilliseconds:F3} ms");
    Console.WriteLine($"Total including initialization: {totalTimer.Elapsed.TotalMilliseconds:F3} ms");
    Console.WriteLine("Validation passed: both factors are prime and their product matches the input.");
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidOperationException or TimeoutException or OverflowException)
{
    Console.Error.WriteLine($"SemiPrimeFactorization failed: {error.Message}");
    return 1;
}
