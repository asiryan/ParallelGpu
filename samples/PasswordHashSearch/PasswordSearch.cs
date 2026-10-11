using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using ParallelGpu;

namespace PasswordHashSearch;

/// <summary>Searches lowercase ASCII passwords within a configured length range using ordinary GPU lambdas.</summary>
public sealed class PasswordSearch
{
    /// <summary>The number of lowercase ASCII letters from 'a' through 'z'.</summary>
    private const int AlphabetSize = 26;

    /// <summary>The borrowed executor used to compile and run the search loop.</summary>
    private readonly Gpu gpu;

    /// <summary>Gets the shortest candidate length searched by this instance.</summary>
    public int MinimumLength { get; }

    /// <summary>Gets the longest candidate length searched by this instance.</summary>
    public int MaximumLength { get; }

    /// <summary>Gets the maximum number of candidates hashed before reading a result and checking the budget.</summary>
    public int BatchSize { get; }

    /// <summary>Gets the total number of lowercase ASCII combinations within the configured range.</summary>
    public long CandidateCount { get; }

    /// <summary>Creates a search with an explicit candidate length range and GPU batch size.</summary>
    /// <param name="gpu">The executor to reuse; the caller retains ownership and disposes it.</param>
    /// <param name="minimumLength">The shortest candidate length, from one through eight.</param>
    /// <param name="maximumLength">The longest candidate length, from the minimum length through eight.</param>
    /// <param name="batchSize">The positive maximum number of candidates in one batch.</param>
    /// <exception cref="ArgumentNullException">The executor is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The length range is invalid or the batch size is not positive.</exception>
    public PasswordSearch(Gpu gpu, int minimumLength, int maximumLength, int batchSize = 1_048_576)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        if (minimumLength < 1 || minimumLength > ShortPasswordSha256.MaximumPasswordLength)
            throw new ArgumentOutOfRangeException(nameof(minimumLength), "Minimum length must be between 1 and 8.");
        if (maximumLength < minimumLength || maximumLength > ShortPasswordSha256.MaximumPasswordLength)
            throw new ArgumentOutOfRangeException(nameof(maximumLength), "Maximum length must be between the minimum length and 8.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        this.gpu = gpu;
        MinimumLength = minimumLength;
        MaximumLength = maximumLength;
        BatchSize = batchSize;
        long total = 0;
        for (int length = minimumLength; length <= maximumLength; length++)
            total += CountCandidates(length);
        CandidateCount = total;
    }

    /// <summary>Returns the number of lowercase ASCII combinations at one supported length.</summary>
    /// <param name="length">The candidate length, validated by the constructor.</param>
    /// <returns>26 raised to the candidate length, using a 64-bit count.</returns>
    private static long CountCandidates(int length)
    {
        long count = 1;
        for (int i = 0; i < length; i++)
            count *= AlphabetSize;
        return count;
    }

    /// <summary>Encodes a valid candidate index as up to eight little-endian ASCII bytes.</summary>
    /// <param name="index">The zero-based position among candidates of this length in lexicographic order.</param>
    /// <param name="length">The candidate length, validated by the constructor.</param>
    /// <returns>The candidate bytes with the first character in the least significant byte.</returns>
    private static ulong PackCandidate(ulong index, int length)
    {
        ulong packed = 0;
        // Base-26 digits map to letters; the rightmost character changes fastest.
        for (int position = length - 1; position >= 0; position--)
        {
            packed |= ('a' + index % AlphabetSize) << (position * 8);
            index /= AlphabetSize;
        }
        return packed;
    }

    /// <summary>Converts a matching global candidate index back to its host string.</summary>
    /// <param name="index">The matching position among candidates of this length.</param>
    /// <param name="length">The candidate length, validated by the constructor.</param>
    /// <returns>The lowercase ASCII password represented by the index.</returns>
    private static string DecodeCandidate(ulong index, int length)
    {
        ulong packed = PackCandidate(index, length);
        var characters = new char[length];
        for (int i = 0; i < length; i++)
            characters[i] = (char)((packed >> (i * 8)) & 0xff);
        return new string(characters);
    }

    /// <summary>Uses the independent framework implementation to create the synthetic target.</summary>
    /// <param name="password">The plaintext to encode as ASCII and hash on the host.</param>
    /// <returns>The SHA-256 digest represented as eight big-endian words.</returns>
    /// <exception cref="ArgumentNullException">The password is null.</exception>
    public static Sha256Digest HashOnHost(string password)
    {
        byte[] hash = SHA256.HashData(Encoding.ASCII.GetBytes(password));
        return new Sha256Digest
        {
            A = BinaryPrimitives.ReadUInt32BigEndian(hash.AsSpan(0, 4)),
            B = BinaryPrimitives.ReadUInt32BigEndian(hash.AsSpan(4, 4)),
            C = BinaryPrimitives.ReadUInt32BigEndian(hash.AsSpan(8, 4)),
            D = BinaryPrimitives.ReadUInt32BigEndian(hash.AsSpan(12, 4)),
            E = BinaryPrimitives.ReadUInt32BigEndian(hash.AsSpan(16, 4)),
            F = BinaryPrimitives.ReadUInt32BigEndian(hash.AsSpan(20, 4)),
            G = BinaryPrimitives.ReadUInt32BigEndian(hash.AsSpan(24, 4)),
            H = BinaryPrimitives.ReadUInt32BigEndian(hash.AsSpan(28, 4))
        };
    }

    /// <summary>Formats a SHA-256 digest as lowercase hexadecimal text.</summary>
    /// <param name="hash">The digest to display.</param>
    /// <returns>The 64-character hexadecimal representation of the digest.</returns>
    public static string FormatHash(Sha256Digest hash) =>
        $"{hash.A:x8}{hash.B:x8}{hash.C:x8}{hash.D:x8}{hash.E:x8}{hash.F:x8}{hash.G:x8}{hash.H:x8}";

    /// <summary>Compiles and launches the same loop body used by the actual search.</summary>
    /// <param name="target">The target digest used by the warm-up iteration.</param>
    /// <remarks>Hashes one candidate and discards the result; subsequent calls reuse the compiled body.</remarks>
    public void WarmUp(Sha256Digest target) => SearchRange(target, MinimumLength, 0, 1);

    /// <summary>Searches increasing lengths in batches, checking the time budget between launches.</summary>
    /// <param name="target">The digest to match; the search does not receive the expected plaintext.</param>
    /// <param name="timeLimit">The positive time budget for this search, including compilation if not warmed up.</param>
    /// <returns>The recovered password or a null password when all candidates have been exhausted.</returns>
    /// <remarks>Gpu.For is synchronous; a running launch or compilation cannot be interrupted.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">The time budget is not positive.</exception>
    /// <exception cref="TimeoutException">The elapsed time reaches the budget before or after a batch.</exception>
    public SearchResult Find(Sha256Digest target, TimeSpan timeLimit)
    {
        if (timeLimit <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(timeLimit));

        var timer = Stopwatch.StartNew();
        long tested = 0;
        for (int length = MinimumLength; length <= MaximumLength; length++)
        {
            long lengthCount = CountCandidates(length);
            for (long start = 0; start < lengthCount;)
            {
                CheckTime(timer, timeLimit);
                int count = (int)Math.Min((long)BatchSize, lengthCount - start);
                int winner = SearchRange(target, length, start, count);
                tested += count;
                CheckTime(timer, timeLimit);
                if (winner != int.MaxValue)
                    return new(DecodeCandidate((ulong)(start + winner), length), tested, timer.Elapsed);
                start += count;
            }
        }
        return new(null, tested, timer.Elapsed);
    }

    /// <summary>Rejects further search work when the elapsed time reaches the budget.</summary>
    /// <param name="timer">The timer measuring this search.</param>
    /// <param name="timeLimit">The maximum permitted elapsed time.</param>
    /// <exception cref="TimeoutException">The timer has reached or exceeded the budget.</exception>
    private static void CheckTime(Stopwatch timer, TimeSpan timeLimit)
    {
        if (timer.Elapsed >= timeLimit)
            throw new TimeoutException("The demo exceeded its time budget between GPU batches.");
    }

    /// <summary>Hashes one candidate per device iteration with automatic compilation and result transfers.</summary>
    /// <param name="target">The digest to match.</param>
    /// <param name="length">The common length of the candidates in this batch.</param>
    /// <param name="start">The inclusive global candidate index.</param>
    /// <param name="count">The number of candidates in this batch.</param>
    /// <returns>The smallest matching batch index, or Int32.MaxValue if the batch contains no match.</returns>
    private int SearchRange(Sha256Digest target, int length, long start, int count)
    {
        int[] winner = [int.MaxValue];
        // Capture scalar settings rather than the search instance in the device body.
        ulong offset = (ulong)start;
        // Local loop indices stay within Int32 even when the global search space is larger.
        gpu.For(0, count, index =>
        {
            ulong packed = PackCandidate(offset + (ulong)index, length);
            Sha256Digest hash = ShortPasswordSha256.Compute(packed, length);
            if (hash.EqualsDigest(target))
                // Shared output requires an atomic write because iterations execute in any order.
                GpuAtomic.Min(ref winner[0], index);
        });
        return winner[0];
    }
}

/// <summary>Contains the recovered plaintext and measurements from the completed GPU search.</summary>
/// <param name="Password">The recovered password, or null if no candidate matched.</param>
/// <param name="CandidatesTested">The number of candidates hashed, including the entire final batch.</param>
/// <param name="Elapsed">The search duration, including launches and transfers.</param>
public readonly record struct SearchResult(string? Password, long CandidatesTested, TimeSpan Elapsed);
