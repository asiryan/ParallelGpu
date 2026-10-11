using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class NumericRegressionFixTests(GpuFixture fixture)
{
    [Fact]
    public void CopySignPreservesMagnitudePayloadAndReplacesOnlyTheSignBit()
    {
        float[] magnitudes32 = [0F, -0F, 1F, -1F, float.Epsilon, float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity,
            BitConverter.Int32BitsToSingle(0x7fc12345), BitConverter.Int32BitsToSingle(unchecked((int)0xffa12345))];
        float[] signs32 = [0F, -0F, 3F, -3F, float.PositiveInfinity, float.NegativeInfinity,
            BitConverter.Int32BitsToSingle(0x7fc00001), BitConverter.Int32BitsToSingle(unchecked((int)0xffc00001))];
        double[] magnitudes64 = [0D, -0D, 1D, -1D, double.Epsilon, double.MaxValue,
            double.PositiveInfinity, double.NegativeInfinity,
            BitConverter.Int64BitsToDouble(0x7ff8123456789abc),
            BitConverter.Int64BitsToDouble(unchecked((long)0xfff0123456789abc))];
        double[] signs64 = [0D, -0D, 3D, -3D, double.PositiveInfinity, double.NegativeInfinity,
            BitConverter.Int64BitsToDouble(0x7ff8000000000001),
            BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000001))];
        float[] values32 = magnitudes32.SelectMany(value => signs32.Select(_ => value)).ToArray();
        float[] inputSigns32 = magnitudes32.SelectMany(_ => signs32).ToArray();
        double[] values64 = magnitudes64.SelectMany(value => signs64.Select(_ => value)).ToArray();
        double[] inputSigns64 = magnitudes64.SelectMany(_ => signs64).ToArray();
        float[] result32 = new float[values32.Length];
        double[] result64 = new double[values64.Length];
        fixture.Gpu.For(0, result32.Length, i =>
        {
            result32[i] = MathF.CopySign(values32[i], inputSigns32[i]);
            result64[i] = CopySignThroughHelper(values64[i], inputSigns64[i]);
        });
        for (int i = 0; i < result32.Length; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(MathF.CopySign(values32[i], inputSigns32[i])),
                BitConverter.SingleToInt32Bits(result32[i]));
            Assert.Equal(BitConverter.DoubleToInt64Bits(Math.CopySign(values64[i], inputSigns64[i])),
                BitConverter.DoubleToInt64Bits(result64[i]));
        }
    }

    [Fact]
    public void FloatingRemainderMatchesClrForSpecialValuesSubnormalsAndLargeQuotients()
    {
        float[] special32 = [0F, -0F, 1F, -1F, 5.2F, -5.2F, float.Epsilon,
            BitConverter.Int32BitsToSingle(3), BitConverter.Int32BitsToSingle(0x007fffff),
            BitConverter.Int32BitsToSingle(0x00800000), float.MaxValue, -float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN];
        double[] special64 = [0D, -0D, 1D, -1D, 5.2D, -5.2D, double.Epsilon,
            BitConverter.Int64BitsToDouble(3), BitConverter.Int64BitsToDouble(0x000fffffffffffff),
            BitConverter.Int64BitsToDouble(0x0010000000000000), double.MaxValue, -double.MaxValue,
            double.PositiveInfinity, double.NegativeInfinity, double.NaN];
        var pairs32 = special32.SelectMany(left => special32.Select(right => (Left: left, Right: right))).ToList();
        var pairs64 = special64.SelectMany(left => special64.Select(right => (Left: left, Right: right))).ToList();
        // A raw-bit corpus samples normal and subnormal values across the whole exponent range.
        var random = new Random(19371);
        byte[] bytes = new byte[8];
        for (int i = 0; i < 2048; i++)
        {
            random.NextBytes(bytes);
            float left32 = BitConverter.Int32BitsToSingle(BitConverter.ToInt32(bytes, 0));
            float right32 = BitConverter.Int32BitsToSingle(BitConverter.ToInt32(bytes, 4));
            double left64 = BitConverter.Int64BitsToDouble(BitConverter.ToInt64(bytes, 0));
            random.NextBytes(bytes);
            double right64 = BitConverter.Int64BitsToDouble(BitConverter.ToInt64(bytes, 0));
            pairs32.Add((left32, right32));
            pairs64.Add((left64, right64));
        }
        float[] left32Values = pairs32.Select(pair => pair.Left).ToArray();
        float[] right32Values = pairs32.Select(pair => pair.Right).ToArray();
        double[] left64Values = pairs64.Select(pair => pair.Left).ToArray();
        double[] right64Values = pairs64.Select(pair => pair.Right).ToArray();
        float[] direct32 = new float[pairs32.Count], helper32 = new float[pairs32.Count];
        double[] direct64 = new double[pairs64.Count], constructor64 = new double[pairs64.Count];
        fixture.Gpu.For(0, direct32.Length, i =>
        {
            direct32[i] = left32Values[i] % right32Values[i];
            helper32[i] = RemainderThroughHelper(left32Values[i], right32Values[i]);
            direct64[i] = left64Values[i] % right64Values[i];
            constructor64[i] = new RemainderValue(left64Values[i], right64Values[i]).Value;
        });
        for (int i = 0; i < direct32.Length; i++)
        {
            float expected32 = left32Values[i] % right32Values[i];
            double expected64 = left64Values[i] % right64Values[i];
            AssertSingle(expected32, direct32[i]);
            AssertSingle(expected32, helper32[i]);
            AssertDouble(expected64, direct64[i]);
            AssertDouble(expected64, constructor64[i]);
        }
    }

    private static double CopySignThroughHelper(double value, double sign) => Math.CopySign(value, sign);
    private static float RemainderThroughHelper(float left, float right) => left % right;
    private readonly struct RemainderValue(double left, double right)
    {
        public readonly double Value = left % right;
    }

    private static void AssertSingle(float expected, float actual)
    {
        if (float.IsNaN(expected))
            Assert.True(float.IsNaN(actual));
        else
            Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual));
    }
    private static void AssertDouble(double expected, double actual)
    {
        if (double.IsNaN(expected))
            Assert.True(double.IsNaN(actual));
        else
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual));
    }
}
