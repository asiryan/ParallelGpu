namespace MonteCarloPi;

/// <summary>Generates deterministic random points directly from their indices without mutable generator state.</summary>
public static class RandomPoints
{
    /// <summary>The odd increment used by the SplitMix64 generator.</summary>
    private const ulong Increment = 0x9e3779b97f4a7c15ul;

    /// <summary>Converts a 53-bit integer to an exactly representable value in [0, 1).</summary>
    private const double UnitScale = 1.0 / 9_007_199_254_740_992.0;

    /// <summary>Generates a point in the unit square and tests whether it lies inside the unit quarter circle.</summary>
    /// <param name="seed">The seed shared by all points in one estimate.</param>
    /// <param name="index">The global point index, less than 2^63; it is independent of GPU work partitioning.</param>
    /// <returns>True when the point satisfies x squared plus y squared less than or equal to one.</returns>
    public static bool IsInsideQuarterCircle(ulong seed, ulong index)
    {
        // Each point consumes two consecutive SplitMix64 values. Arithmetic intentionally wraps modulo 2^64.
        // The pair sequence repeats after 2^63 points; this sample accepts a positive signed 64-bit count.
        unchecked
        {
            ulong counter = seed + (index * 2 + 1) * Increment;
            double x = (Mix(counter) >> 11) * UnitScale;
            double y = (Mix(counter + Increment) >> 11) * UnitScale;
            return x * x + y * y <= 1.0;
        }
    }

    /// <summary>Scrambles a counter with the SplitMix64 mixing function using only device-compatible arithmetic.</summary>
    /// <param name="value">The counter to mix.</param>
    /// <returns>A deterministic pseudorandom 64-bit value.</returns>
    private static ulong Mix(ulong value)
    {
        unchecked
        {
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
            return value ^ (value >> 31);
        }
    }
}
