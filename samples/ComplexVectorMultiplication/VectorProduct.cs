using System.Numerics;
using ParallelGpu;

namespace ComplexVectorMultiplication;

/// <summary>Computes sum(a[i] * b[i]) without conjugating either vector.</summary>
public static class VectorProduct
{
    /// <summary>The number of adjacent products combined by each reduction iteration on either backend.</summary>
    internal const int BatchSize = 256;

    /// <summary>Multiplies elements with ForEach and reduces their sum entirely on the selected device.</summary>
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

        using var products = gpu.Upload(a);
        using var right = gpu.Upload(b);
        gpu.ForEach(products, new MultiplyComplexKernel(right.View));
        return ReduceOnGpu(gpu, products);
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

        var products = new Complex[length];
        Parallel.For(0, length, i => products[i] = a[i] * b[i]);
        return ReduceOnCpu(products);
    }

    /// <summary>Combines adjacent device values until only the final sum remains.</summary>
    /// <param name="gpu">The executor owning the reduction buffers.</param>
    /// <param name="values">A nonempty device buffer containing products or partial sums.</param>
    /// <returns>The single complex result downloaded after the last reduction pass.</returns>
    private static Complex ReduceOnGpu(Gpu gpu, GpuBuffer<Complex> values)
    {
        if (values.Length == 1)
            return values.ToArray()[0];

        using var sums = gpu.Allocate<Complex>(BatchCount(values.Length));
        gpu.For(0, sums.Length, new SumComplexKernel(values.View, sums.View));
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
