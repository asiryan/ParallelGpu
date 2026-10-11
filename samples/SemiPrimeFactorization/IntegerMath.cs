using System.Numerics;

namespace SemiPrimeFactorization;

/// <summary>Provides host arithmetic for the search boundary and validation of the recovered factors.</summary>
public static class IntegerMath
{
    /// <summary>The first twelve prime bases used for deterministic Miller-Rabin validation.</summary>
    private static readonly int[] PrimeBases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    /// <summary>Returns the greatest integer whose square is no larger than a positive value.</summary>
    /// <param name="value">The positive integer whose square root is required.</param>
    /// <returns>The exact integer floor of the square root.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The value is not positive.</exception>
    public static long FloorSquareRoot(long value)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);

        long root = (long)Math.Sqrt(value);
        // Correct floating-point rounding with division so the comparison cannot overflow.
        while (root > value / root)
            root--;
        while (root + 1 <= value / (root + 1))
            root++;
        return root;
    }

    /// <summary>Determines whether a signed 64-bit integer is prime.</summary>
    /// <param name="value">The integer to validate; values below two are not prime.</param>
    /// <returns>True exactly when the value is prime.</returns>
    /// <remarks>
    /// The first twelve prime bases are deterministic below 318665857834031151167461,
    /// which covers every positive Int64 value; see
    /// <see href="https://arxiv.org/abs/1509.00864">Sorenson and Webster</see>.
    /// This host check validates results and does not search for factors.
    /// </remarks>
    public static bool IsPrime(long value)
    {
        if (value < 2)
            return false;
        foreach (int prime in PrimeBases)
        {
            if (value == prime)
                return true;
            if (value % prime == 0)
                return false;
        }

        // Miller-Rabin writes value - 1 as an odd number times a power of two.
        long oddPart = value - 1;
        int powersOfTwo = 0;
        while ((oddPart & 1) == 0)
        {
            oddPart >>= 1;
            powersOfTwo++;
        }

        BigInteger modulus = value;
        foreach (int prime in PrimeBases)
        {
            // BigInteger keeps modular multiplication exact near Int64.MaxValue.
            BigInteger residue = BigInteger.ModPow(prime, oddPart, modulus);
            if (residue == BigInteger.One || residue == modulus - 1)
                continue;

            bool reachedMinusOne = false;
            for (int power = 1; power < powersOfTwo; power++)
            {
                residue = residue * residue % modulus;
                if (residue == modulus - 1)
                {
                    reachedMinusOne = true;
                    break;
                }
            }
            if (!reachedMinusOne)
                return false;
        }
        return true;
    }
}
