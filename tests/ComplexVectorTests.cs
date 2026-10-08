using System.Numerics;
using ComplexVectorMultiplication;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ComplexVectorTests(GpuFixture fixture)
{
    [Fact]
    public void CpuAndGpuMatchKnownProductsAndReductionTails()
    {
        Complex[] a = [new(1, 2), new(-3, 4)];
        Complex[] b = [new(5, 6), new(7, -8)];
        Assert.Equal(new Complex(4, 68), VectorProduct.MultiplyOnCpu(a, b));
        Assert.Equal(new Complex(4, 68), VectorProduct.MultiplyOnGpu(fixture.Gpu, a, b));
        Assert.Equal(new Complex[] { new(1, 2), new(-3, 4) }, a);
        Assert.Equal(new Complex[] { new(5, 6), new(7, -8) }, b);

        // Cover batch groups with 1, 2, 31 and 32 lanes, incomplete groups, multiple
        // reduction passes, and arrays large enough to use direct device transfers.
        foreach (int length in new[] { 0, 1, 255, 257, 7_935, 7_937, 8_192, 8_193, 8_449, 65_537, 131_073 })
        {
            a = Vectors.Random(length, seed: 31);
            b = Vectors.Random(length, seed: 47);
            Complex[] originalA = (Complex[])a.Clone();
            Complex[] originalB = (Complex[])b.Clone();
            Complex expected = Complex.Zero;
            for (int i = 0; i < length; i++)
                expected += a[i] * b[i];

            Complex cpu = VectorProduct.MultiplyOnCpu(a, b);
            Complex gpu = VectorProduct.MultiplyOnGpu(fixture.Gpu, a, b);
            Vectors.CheckEqual(expected, cpu);
            Vectors.CheckEqual(expected, gpu);
            Assert.Equal(originalA, a);
            Assert.Equal(originalB, b);
        }
    }
}
