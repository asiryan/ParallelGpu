using System.Diagnostics;
using ParallelGpu;
using PasswordHashSearch;

// Set the demo password and candidate length range here. The search receives only the hash.
const string DemoPassword = "qwerty";
const int MinimumPasswordLength = 1;
const int MaximumPasswordLength = 6;
const int BatchSize = 1_048_576;
TimeSpan budget = TimeSpan.FromSeconds(30);

try
{
    foreach (char character in DemoPassword)
        if (character < 'a' || character > 'z')
            throw new ArgumentException("DemoPassword must contain only lowercase letters a-z.");

    // Include device initialization and kernel compilation in the total runtime budget.
    var totalTimer = Stopwatch.StartNew();

    // Create the synthetic target on the host; the search receives only this digest.
    Sha256Digest target = PasswordSearch.HashOnHost(DemoPassword);
    Console.WriteLine($"Synthetic password recovery: unsalted SHA-256, lowercase a-z, length {MinimumPasswordLength}..{MaximumPasswordLength}");
    Console.WriteLine($"Target SHA-256: {PasswordSearch.FormatHash(target)}");

    // Require a real GPU. Auto tries CUDA first, then compatible OpenCL devices.
    using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
    var search = new PasswordSearch(gpu,
        minimumLength: MinimumPasswordLength,
        maximumLength: MaximumPasswordLength,
        batchSize: BatchSize);
    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Search space: {search.CandidateCount:N0} candidates; total runtime budget: {budget.TotalSeconds:G} seconds");

    // Measure kernel warm-up separately while keeping it inside the overall budget.
    search.WarmUp(target);
    TimeSpan setup = totalTimer.Elapsed;
    if (setup >= budget)
        throw new TimeoutException("Device initialization and kernel compilation exceeded the runtime budget.");
    Console.WriteLine($"Initialization and kernel warm-up: {setup.TotalMilliseconds:F3} ms");

    // Pass only the remaining time after initialization to the batched search.
    SearchResult result = search.Find(target, budget - totalTimer.Elapsed);
    if (result.Password is null)
        throw new InvalidOperationException("The synthetic password was not found within the configured length range.");
    // Validate the recovered plaintext using the independent framework implementation.
    if (result.Password != DemoPassword ||
        !PasswordSearch.HashOnHost(result.Password).EqualsDigest(target))
        throw new InvalidOperationException("Independent .NET SHA-256 validation failed.");

    totalTimer.Stop();
    if (totalTimer.Elapsed >= budget)
        throw new TimeoutException("The complete demonstration exceeded the runtime budget.");

    Console.WriteLine($"Found password: {result.Password}");
    Console.WriteLine($"Candidates hashed: {result.CandidatesTested:N0}");
    Console.WriteLine($"GPU search, including transfers: {result.Elapsed.TotalMilliseconds:F3} ms");
    Console.WriteLine($"Throughput: {result.CandidatesTested / result.Elapsed.TotalSeconds:N0} hashes/s");
    Console.WriteLine($"Total including initialization: {totalTimer.Elapsed.TotalMilliseconds:F3} ms");
    Console.WriteLine("Validation passed: recovered plaintext matches the framework SHA-256 hash.");
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidOperationException or TimeoutException)
{
    Console.Error.WriteLine($"PasswordHashSearch failed: {error.Message}");
    return 1;
}
