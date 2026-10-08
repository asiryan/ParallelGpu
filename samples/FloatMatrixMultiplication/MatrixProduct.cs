using ParallelGpu;

namespace FloatMatrixMultiplication;

/// <summary>Computes compatible matrix products on the CPU or a selected accelerator.</summary>
public static class MatrixProduct
{
    /// <summary>Computes one output element per device iteration with automatic compilation and transfers.</summary>
    /// <param name="gpu">The executor used to compile and run the matrix multiplication loop.</param>
    /// <param name="a">The left matrix, which remains unchanged.</param>
    /// <param name="b">The right matrix, which remains unchanged.</param>
    /// <returns>A new matrix with A's row count and B's column count containing their product.</returns>
    /// <exception cref="ArgumentNullException">The executor or either matrix is null.</exception>
    /// <exception cref="ArgumentException">A matrix is empty or ragged, or the inner dimensions differ.</exception>
    /// <exception cref="OverflowException">The output element count exceeds Int32.MaxValue.</exception>
    public static float[][] MultiplyOnGpu(Gpu gpu, float[][] a, float[][] b)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        var (rows, inner, columns) = Matrices.ProductShape(a, b);
        float[][] result = Matrices.Create(rows, columns);

        gpu.For<float>(0, rows * columns, index =>
        {
            int row = index / columns;
            int column = index % columns;
            var leftRow = a[row];
            float sum = 0;
            for (int k = 0; k < inner; k++)
                sum += leftRow[k] * b[k][column];
            result[row][column] = sum;
        });

        return result;
    }

    /// <summary>Computes one output element per Parallel.For iteration using the same summation order as the GPU.</summary>
    /// <param name="a">The left matrix, which remains unchanged.</param>
    /// <param name="b">The right matrix, which remains unchanged.</param>
    /// <returns>A new matrix with A's row count and B's column count containing their product.</returns>
    /// <exception cref="ArgumentNullException">Either matrix is null.</exception>
    /// <exception cref="ArgumentException">A matrix is empty or ragged, or the inner dimensions differ.</exception>
    /// <exception cref="OverflowException">The output element count exceeds Int32.MaxValue.</exception>
    public static float[][] MultiplyOnCpu(float[][] a, float[][] b)
    {
        var (rows, inner, columns) = Matrices.ProductShape(a, b);
        float[][] result = Matrices.Create(rows, columns);

        Parallel.For(0, rows * columns, index =>
        {
            int row = index / columns;
            int column = index % columns;
            var leftRow = a[row];
            float sum = 0;
            for (int k = 0; k < inner; k++)
                sum += leftRow[k] * b[k][column];
            result[row][column] = sum;
        });

        return result;
    }
}
