using ParallelGpu;

namespace MatrixMultiplication;

public static class MatrixProduct
{
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

    public static float[][] MultiplyOnCpu(float[][] a, float[][] b)
    {
        var (rows, inner, columns) = Matrices.ProductShape(a, b);
        float[][] result = Matrices.Create(rows, columns);

        Parallel.For(0, rows, row =>
        {
            for (int column = 0; column < columns; column++)
            {
                float sum = 0;
                for (int k = 0; k < inner; k++)
                    sum += a[row][k] * b[k][column];
                result[row][column] = sum;
            }
        });

        return result;
    }
}
