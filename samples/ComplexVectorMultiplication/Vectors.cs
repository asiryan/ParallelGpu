using System.Numerics;

namespace ComplexVectorMultiplication;

/// <summary>Creates complex vectors and validates their product inputs and results.</summary>
public static class Vectors
{
    /// <summary>Creates reproducible complex values with real and imaginary components in [-1, 1).</summary>
    /// <param name="length">The nonnegative number of vector elements.</param>
    /// <param name="seed">The random generator seed.</param>
    /// <returns>A vector containing the generated values.</returns>
    public static Complex[] Random(int length, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        var values = new Complex[length];
        var random = new Random(seed);
        for (int i = 0; i < values.Length; i++)
            values[i] = new Complex(random.NextDouble() * 2 - 1, random.NextDouble() * 2 - 1);
        return values;
    }

    /// <summary>Validates both components with an absolute and relative tolerance.</summary>
    /// <param name="expected">The CPU reference result.</param>
    /// <param name="actual">The GPU result.</param>
    /// <returns>The maximum absolute component error.</returns>
    /// <exception cref="InvalidOperationException">A component is nonfinite or differs beyond tolerance.</exception>
    public static double CheckEqual(Complex expected, Complex actual)
    {
        double realError = Math.Abs(actual.Real - expected.Real);
        double imaginaryError = Math.Abs(actual.Imaginary - expected.Imaginary);
        if (!double.IsFinite(expected.Real) || !double.IsFinite(expected.Imaginary) ||
            !double.IsFinite(actual.Real) || !double.IsFinite(actual.Imaginary) ||
            realError > 1e-10 + 1e-10 * Math.Abs(expected.Real) ||
            imaginaryError > 1e-10 + 1e-10 * Math.Abs(expected.Imaginary))
            throw new InvalidOperationException($"Results differ: {expected} vs {actual}.");
        return Math.Max(realError, imaginaryError);
    }

    /// <summary>Validates the vectors and returns their common length.</summary>
    /// <param name="a">The left vector.</param>
    /// <param name="b">The right vector.</param>
    /// <returns>The number of products to compute.</returns>
    /// <exception cref="ArgumentNullException">A vector is null.</exception>
    /// <exception cref="ArgumentException">The vector lengths differ.</exception>
    internal static int ProductLength(Complex[] a, Complex[] b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match.", nameof(b));
        return a.Length;
    }
}
