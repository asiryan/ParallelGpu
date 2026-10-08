using System.Numerics;
using ParallelGpu;

namespace ComplexVectorMultiplication;

/// <summary>Computes sum(a[i] * b[i]) without conjugating either vector.</summary>
public static class VectorProduct
{
    /// <summary>The maximum number of products combined by each iteration on either backend.</summary>
    internal const int BatchSize = 256;

    /// <summary>Neighboring iterations read neighboring elements within each group of batches.</summary>
    private const int BatchLanes = 32;

    /// <summary>Combines multiplication and the first reduction pass with automatic array transfers.</summary>
    /// <param name="gpu">The executor used for multiplication and reduction.</param>
    /// <param name="a">The left vector, which remains unchanged.</param>
    /// <param name="b">The right vector, which remains unchanged.</param>
    /// <returns>The complex sum of the element products, or zero for empty vectors.</returns>
    public static Complex MultiplyOnGpu(Gpu gpu, Complex[] a, Complex[] b)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        int length = Vectors.ProductLength(a, b);
        if (length == 0)
            return Complex.Zero;

        var sums = new Complex[BatchCount(length)];
        gpu.For<Complex>(0, sums.Length, index =>
        {
            // Interleave each group so neighboring GPU threads load neighboring elements.
            int group = index / BatchLanes * BatchLanes;
            int stride = Math.Min(BatchLanes, sums.Length - group);
            int start = group * BatchSize + index - group;
            int count = Math.Min(BatchSize, 1 + (length - 1 - start) / stride);
            Complex sum = Complex.Zero;
            for (int i = 0; i < count; i++)
                sum += a[start + i * stride] * b[start + i * stride];
            sums[index] = sum;
        });
        return ReduceOnGpu(gpu, sums);
    }

    /// <summary>Multiplies elements and reduces their sum using the same batch order as the GPU.</summary>
    /// <param name="a">The left vector, which remains unchanged.</param>
    /// <param name="b">The right vector, which remains unchanged.</param>
    /// <returns>The complex sum of the element products, or zero for empty vectors.</returns>
    public static Complex MultiplyOnCpu(Complex[] a, Complex[] b)
    {
        int length = Vectors.ProductLength(a, b);
        if (length == 0)
            return Complex.Zero;

        var sums = new Complex[BatchCount(length)];
        Parallel.For(0, sums.Length, index =>
        {
            // Interleave each group so neighboring GPU threads load neighboring elements.
            int group = index / BatchLanes * BatchLanes;
            int stride = Math.Min(BatchLanes, sums.Length - group);
            int start = group * BatchSize + index - group;
            int count = Math.Min(BatchSize, 1 + (length - 1 - start) / stride);
            Complex sum = Complex.Zero;
            for (int i = 0; i < count; i++)
                sum += a[start + i * stride] * b[start + i * stride];
            sums[index] = sum;
        });
        return ReduceOnCpu(sums);
    }

    /// <summary>Combines adjacent values until only the final sum remains, transferring arrays automatically.</summary>
    /// <param name="gpu">The executor running the reduction loops.</param>
    /// <param name="values">A nonempty array containing products or partial sums.</param>
    /// <returns>The single complex result after the last reduction pass.</returns>
    private static Complex ReduceOnGpu(Gpu gpu, Complex[] values)
    {
        if (values.Length == 1)
            return values[0];

        var sums = new Complex[BatchCount(values.Length)];
        gpu.For<Complex>(0, sums.Length, index =>
        {
            int start = index * BatchSize;
            int count = Math.Min(BatchSize, values.Length - start);
            Complex sum = Complex.Zero;
            for (int i = 0; i < count; i++)
                sum += values[start + i];
            sums[index] = sum;
        });
        return ReduceOnGpu(gpu, sums);
    }

    /// <summary>Combines adjacent host values in the same order as the device reduction.</summary>
    /// <param name="values">A nonempty array containing products or partial sums.</param>
    /// <returns>The single complex result after the last reduction pass.</returns>
    private static Complex ReduceOnCpu(Complex[] values)
    {
        if (values.Length == 1)
            return values[0];

        var sums = new Complex[BatchCount(values.Length)];
        Parallel.For(0, sums.Length, index =>
        {
            int start = index * BatchSize;
            int count = Math.Min(BatchSize, values.Length - start);
            Complex sum = Complex.Zero;
            for (int i = 0; i < count; i++)
                sum += values[start + i];
            sums[index] = sum;
        });
        return ReduceOnCpu(sums);
    }

    /// <summary>Returns the number of batches, including the final incomplete batch.</summary>
    /// <param name="length">The positive number of values to reduce.</param>
    /// <returns>The number of required partial sums.</returns>
    private static int BatchCount(int length) => 1 + (length - 1) / BatchSize;
}
