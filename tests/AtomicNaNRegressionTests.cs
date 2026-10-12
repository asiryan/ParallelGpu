using ILGPU;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AtomicNaNRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void FloatingMinMaxReturnThePreviousNaNPayload()
    {
        float nan32 = BitConverter.Int32BitsToSingle(0x7fc12345);
        double nan64 = BitConverter.Int64BitsToDouble(0x7ff8123456789abc);
        float[] minimum32 = [nan32], maximum32 = [nan32], previous32 = new float[2];
        double[] minimum64 = [nan64], maximum64 = [nan64], previous64 = new double[2];
        fixture.Gpu.For(0, 1, i =>
        {
            previous32[0] = GpuAtomic.Min(ref minimum32[0], 1f);
            previous32[1] = GpuAtomic.Max(ref maximum32[0], 1f);
            previous64[0] = GpuAtomic.Min(ref minimum64[0], 1d);
            previous64[1] = GpuAtomic.Max(ref maximum64[0], 1d);
        });
        Assert.All(previous32, value => Assert.Equal(BitConverter.SingleToInt32Bits(nan32), BitConverter.SingleToInt32Bits(value)));
        Assert.All(previous64, value => Assert.Equal(BitConverter.DoubleToInt64Bits(nan64), BitConverter.DoubleToInt64Bits(value)));
    }

    [Fact]
    public void FloatingCompareExchangeMatchesNaNPayloadAndSignedZeroBits()
    {
        float nan32 = BitConverter.Int32BitsToSingle(0x7fc12345);
        double nan64 = BitConverter.Int64BitsToDouble(0x7ff8123456789abc);
        float otherNan32 = BitConverter.Int32BitsToSingle(0x7fc12346);
        double otherNan64 = BitConverter.Int64BitsToDouble(0x7ff8123456789abd);
        float[] targets32 = [nan32, nan32, 0.0f, -0.0f, 0.0f, -0.0f];
        double[] targets64 = [nan64, nan64, 0.0, -0.0, 0.0, -0.0];
        float[] comparands32 = [nan32, otherNan32, -0.0f, 0.0f, 0.0f, -0.0f];
        double[] comparands64 = [nan64, otherNan64, -0.0, 0.0, 0.0, -0.0];
        float[] expectedPrevious32 = targets32.ToArray();
        double[] expectedPrevious64 = targets64.ToArray();
        float[] previous32 = new float[targets32.Length];
        double[] previous64 = new double[targets64.Length];
        fixture.Gpu.For(0, targets32.Length, i =>
        {
            previous32[i] = GpuAtomic.CompareExchange(ref targets32[i], 7f, comparands32[i]);
            previous64[i] = GpuAtomic.CompareExchange(ref targets64[i], 7d, comparands64[i]);
        });
        for (int i = 0; i < targets32.Length; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(expectedPrevious32[i]), BitConverter.SingleToInt32Bits(previous32[i]));
            Assert.Equal(BitConverter.DoubleToInt64Bits(expectedPrevious64[i]), BitConverter.DoubleToInt64Bits(previous64[i]));
            Assert.Equal(BitConverter.SingleToInt32Bits(i is 0 or 4 or 5 ? 7f : expectedPrevious32[i]),
                BitConverter.SingleToInt32Bits(targets32[i]));
            Assert.Equal(BitConverter.DoubleToInt64Bits(i is 0 or 4 or 5 ? 7d : expectedPrevious64[i]),
                BitConverter.DoubleToInt64Bits(targets64[i]));
        }
    }

    [Fact]
    public void FloatingAddsCompleteWhenNaNIsAlreadyPresentOrProducedByAnotherIteration()
    {
        float[] nan32 = [float.NaN], becomesNaN32 = [float.PositiveInfinity];
        double[] nan64 = [double.NaN], becomesNaN64 = [double.PositiveInfinity];
        float[] previous32 = new float[32];
        double[] previous64 = new double[32];
        fixture.Gpu.For(0, previous32.Length, i =>
        {
            GpuAtomic.Add(ref nan32[0], 1f);
            GpuAtomic.Add(ref nan64[0], 1d);
            previous32[i] = GpuAtomic.Add(ref becomesNaN32[0], float.NegativeInfinity);
            previous64[i] = GpuAtomic.Add(ref becomesNaN64[0], double.NegativeInfinity);
        });
        Assert.True(float.IsNaN(nan32[0]));
        Assert.True(double.IsNaN(nan64[0]));
        Assert.True(float.IsNaN(becomesNaN32[0]));
        Assert.True(double.IsNaN(becomesNaN64[0]));
        Assert.Equal(1, previous32.Count(float.IsPositiveInfinity));
        Assert.Equal(1, previous64.Count(double.IsPositiveInfinity));
        Assert.Equal(31, previous32.Count(float.IsNaN));
        Assert.Equal(31, previous64.Count(double.IsNaN));
    }

    [Fact]
    public void FloatingMinMaxRemainAtomicUnderContentionAndReturnSignedZeroBits()
    {
        float negativeZero32 = BitConverter.Int32BitsToSingle(int.MinValue);
        double negativeZero64 = BitConverter.Int64BitsToDouble(long.MinValue);
        float[] minimum32 = [1000f], maximum32 = [-1000f], zero32 = [negativeZero32], oldZero32 = new float[1];
        double[] minimum64 = [1000d], maximum64 = [-1000d], zero64 = [negativeZero64], oldZero64 = new double[1];
        fixture.Gpu.For(0, 257, i =>
        {
            GpuAtomic.Min(ref minimum32[0], i - 128f);
            GpuAtomic.Max(ref maximum32[0], i - 128f);
            GpuAtomic.Min(ref minimum64[0], i - 128d);
            GpuAtomic.Max(ref maximum64[0], i - 128d);
        });
        fixture.Gpu.For(0, 1, i =>
        {
            oldZero32[0] = GpuAtomic.Max(ref zero32[0], 0f);
            oldZero64[0] = GpuAtomic.Max(ref zero64[0], 0d);
        });
        Assert.Equal(-128f, minimum32[0]);
        Assert.Equal(128f, maximum32[0]);
        Assert.Equal(-128d, minimum64[0]);
        Assert.Equal(128d, maximum64[0]);
        Assert.Equal(int.MinValue, BitConverter.SingleToInt32Bits(oldZero32[0]));
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(oldZero64[0]));
    }

    [Fact]
    public void ExplicitKernelsPreservePreviousNaNsForAddMinAndMax()
    {
        float nan32 = BitConverter.Int32BitsToSingle(0x7fc12345);
        double nan64 = BitConverter.Int64BitsToDouble(0x7ff8123456789abc);
        using var targets32 = fixture.Gpu.Upload(new[] { nan32, nan32, nan32 });
        using var targets64 = fixture.Gpu.Upload(new[] { nan64, nan64, nan64 });
        using var previous32 = fixture.Gpu.Allocate<float>(3);
        using var previous64 = fixture.Gpu.Allocate<double>(3);
        fixture.Gpu.For(0, 3, new FloatingAtomicSpecialKernel(targets32.View, targets64.View,
            previous32.View, previous64.View));
        Assert.All(previous32.ToArray(), value => Assert.Equal(BitConverter.SingleToInt32Bits(nan32), BitConverter.SingleToInt32Bits(value)));
        Assert.All(previous64.ToArray(), value => Assert.Equal(BitConverter.DoubleToInt64Bits(nan64), BitConverter.DoubleToInt64Bits(value)));
    }
}

public readonly struct FloatingAtomicSpecialKernel(ArrayView<float> targets32, ArrayView<double> targets64,
    ArrayView<float> previous32, ArrayView<double> previous64) : IGpuForKernel
{
    public void Execute(int index)
    {
        previous32[index] = index == 0 ? GpuAtomic.Add(ref targets32[index], 1f) : index == 1
            ? GpuAtomic.Min(ref targets32[index], 1f) : GpuAtomic.Max(ref targets32[index], 1f);
        previous64[index] = index == 0 ? GpuAtomic.Add(ref targets64[index], 1d) : index == 1
            ? GpuAtomic.Min(ref targets64[index], 1d) : GpuAtomic.Max(ref targets64[index], 1d);
    }
}
