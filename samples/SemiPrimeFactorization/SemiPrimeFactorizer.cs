using System.Diagnostics;
using ParallelGpu;

namespace SemiPrimeFactorization;

/// <summary>Finds the two prime factors of a positive Int64 number using GPU trial division.</summary>
public sealed class SemiPrimeFactorizer
{
    /// <summary>The borrowed GPU executor used to test candidate divisors.</summary>
    private readonly Gpu gpu;

    /// <summary>Gets the positive number to factor.</summary>
    public long Number { get; }

    /// <summary>Gets the largest divisor to test, equal to the exact floor of the square root.</summary>
    public long MaximumDivisor { get; }

    /// <summary>Gets the number of candidates, including 2, 3, and eligible values of the form 6k minus or plus 1.</summary>
    public long CandidateCount { get; }

    /// <summary>Gets the maximum number of divisor checks per GPU batch.</summary>
    public int BatchSize { get; }

    /// <summary>Creates a GPU factorizer with an explicit number and batch size.</summary>
    /// <param name="gpu">The GPU executor to reuse; the caller retains ownership and disposes it.</param>
    /// <param name="number">The positive Int64 number, at least four, expected to be a product of two primes.</param>
    /// <param name="batchSize">The positive maximum number of candidate divisors in one batch.</param>
    /// <exception cref="ArgumentNullException">The executor is null.</exception>
    /// <exception cref="ArgumentException">The executor uses a CPU accelerator.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The number is less than four or the batch size is not positive.</exception>
    public SemiPrimeFactorizer(Gpu gpu, long number, int batchSize = 1_048_576)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        if (!gpu.IsGpu)
            throw new ArgumentException("SemiPrimeFactorization requires a GPU accelerator.", nameof(gpu));
        if (number < 4)
            throw new ArgumentOutOfRangeException(nameof(number), "The number must be at least 4.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        this.gpu = gpu;
        Number = number;
        BatchSize = batchSize;
        MaximumDivisor = IntegerMath.FloorSquareRoot(number);
        // Count 2 and 3 separately, then count 6k-1 and 6k+1 up to the exact boundary.
        CandidateCount = 1 + (MaximumDivisor >= 3 ? 1 : 0) +
            (MaximumDivisor + 1) / 6 + (MaximumDivisor - 1) / 6;
    }

    /// <summary>Compiles and launches the same divisor loop used by the actual factor search.</summary>
    /// <remarks>Checks one candidate and discards the result; later calls reuse the compiled body.</remarks>
    public void WarmUp() => SearchRange(0, 1);

    /// <summary>Tests increasing candidate divisors on the GPU and verifies the resulting prime pair.</summary>
    /// <param name="timeLimit">The positive time budget for the search, transfers, and factor validation.</param>
    /// <returns>The two prime factors, number of tested candidates, and elapsed search time.</returns>
    /// <remarks>Time is checked between synchronous batches; an active device call cannot be interrupted.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">The time budget is not positive.</exception>
    /// <exception cref="InvalidOperationException">The number is prime or is not a product of exactly two primes.</exception>
    /// <exception cref="TimeoutException">The elapsed time reaches the budget before or after a batch or validation.</exception>
    public FactorizationResult Factor(TimeSpan timeLimit)
    {
        if (timeLimit <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(timeLimit));

        var timer = Stopwatch.StartNew();
        long tested = 0;
        for (long start = 0; start < CandidateCount;)
        {
            CheckTime(timer, timeLimit);
            int count = (int)Math.Min((long)BatchSize, CandidateCount - start);
            int winner = SearchRange(start, count);
            tested += count;
            CheckTime(timer, timeLimit);
            if (winner != int.MaxValue)
            {
                long first = DivisorAt(start + winner);
                long second = Number / first;
                // Host primality checks validate the GPU result; candidate divisor search stays on the GPU.
                if (Number % first != 0 || !IntegerMath.IsPrime(first) || !IntegerMath.IsPrime(second))
                    throw new InvalidOperationException("The number is not a product of exactly two primes.");
                CheckTime(timer, timeLimit);
                return new(first, second, tested, timer.Elapsed);
            }
            start += count;
        }
        throw new InvalidOperationException("The number is prime and has no nontrivial factor pair.");
    }

    /// <summary>Rejects further work when the elapsed time reaches the configured budget.</summary>
    /// <param name="timer">The timer measuring this factor search.</param>
    /// <param name="timeLimit">The maximum permitted elapsed time.</param>
    /// <exception cref="TimeoutException">The timer has reached or exceeded the budget.</exception>
    private static void CheckTime(Stopwatch timer, TimeSpan timeLimit)
    {
        if (timer.Elapsed >= timeLimit)
            throw new TimeoutException("The factor search exceeded its time budget between GPU batches.");
    }

    /// <summary>Tests one candidate divisor per device iteration with automatic result transfers.</summary>
    /// <param name="start">The inclusive global candidate index.</param>
    /// <param name="count">The number of candidates in this batch.</param>
    /// <returns>The smallest matching local batch index, or Int32.MaxValue if no divisor matched.</returns>
    private int SearchRange(long start, int count)
    {
        int[] winner = [int.MaxValue];
        // Capture scalar values rather than the factorizer instance in the GPU body.
        long number = Number;
        gpu.For(0, count, index =>
        {
            long divisor = DivisorAt(start + index);
            if (number % divisor == 0)
                GpuAtomic.Min(ref winner[0], index);
        });
        return winner[0];
    }

    /// <summary>Maps increasing candidate indices to 2, 3, 5, 7, 11, 13, and subsequent eligible divisors.</summary>
    /// <param name="index">The nonnegative candidate index within the configured search space.</param>
    /// <returns>The corresponding divisor as an Int64 value.</returns>
    private static long DivisorAt(long index)
    {
        if (index == 0)
            return 2;
        if (index == 1)
            return 3;
        long wheelIndex = index - 2;
        return (wheelIndex >> 1) * 6 + 5 + (wheelIndex & 1) * 2;
    }
}

/// <summary>Contains the validated prime factor pair and measurements from the completed GPU search.</summary>
/// <param name="FirstFactor">The smaller prime factor.</param>
/// <param name="SecondFactor">The larger prime factor, which may equal the first.</param>
/// <param name="CandidatesTested">The number of candidate divisors tested, including the entire final batch.</param>
/// <param name="Elapsed">The search duration, including transfers and host factor validation.</param>
public readonly record struct FactorizationResult(long FirstFactor, long SecondFactor,
    long CandidatesTested, TimeSpan Elapsed);
