using MatrixMultiplication;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class MatrixTests(GpuFixture fixture)
{
    [Fact]
    public void CpuAndGpuMatchAKnownRectangularProduct()
    {
        float[][] a = [[1, 2, 3], [4, 5, 6]];
        float[][] b = [[7, 8], [9, 10], [11, 12]];
        float[][] expected = [[58, 64], [139, 154]];

        float[][] cpu = MatrixProduct.MultiplyOnCpu(a, b);
        float[][] gpu = MatrixProduct.MultiplyOnGpu(fixture.Gpu, a, b);

        for (int row = 0; row < expected.Length; row++)
        {
            Assert.Equal(expected[row], cpu[row]);
            Assert.Equal(expected[row], gpu[row]);
        }
        Assert.Equal(new float[] { 1, 2, 3 }, a[0]);
        Assert.Equal(new float[] { 7, 8 }, b[0]);
    }

    [Fact]
    public void CpuAndGpuAgreeWithADoublePrecisionReference()
    {
        float[][] a = Matrices.Random(17, 33, seed: 31);
        float[][] b = Matrices.Random(33, 65, seed: 47);
        float[][] cpu = MatrixProduct.MultiplyOnCpu(a, b);
        float[][] gpu = MatrixProduct.MultiplyOnGpu(fixture.Gpu, a, b);

        for (int row = 0; row < a.Length; row++)
        {
            for (int column = 0; column < b[0].Length; column++)
            {
                double expected = 0;
                for (int k = 0; k < b.Length; k++)
                    expected += (double)a[row][k] * b[k][column];

                double tolerance = 1e-5 + 1e-5 * Math.Abs(expected);
                Assert.InRange(Math.Abs(cpu[row][column] - expected), 0, tolerance);
                Assert.InRange(Math.Abs(gpu[row][column] - expected), 0, tolerance);
            }
        }
    }
}
