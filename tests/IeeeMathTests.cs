using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class IeeeMathTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AdjacentValuesAndBinaryExponentsPreserveExceptionalAndSubnormalBits(bool helper)
    {
        double[] values64 = [double.NaN, double.NegativeInfinity, double.PositiveInfinity, -0.0, 0.0,
            -double.Epsilon, double.Epsilon, double.MinValue, double.MaxValue,
            BitConverter.Int64BitsToDouble(0x000fffffffffffff), BitConverter.Int64BitsToDouble(0x0010000000000000),
            -1, 1, 2, 3];
        float[] values32 = [float.NaN, float.NegativeInfinity, float.PositiveInfinity, -0.0f, 0.0f,
            -float.Epsilon, float.Epsilon, float.MinValue, float.MaxValue,
            BitConverter.Int32BitsToSingle(0x007fffff), BitConverter.Int32BitsToSingle(0x00800000), -1, 1, 2, 3];
        double[] next64 = new double[values64.Length], previous64 = new double[values64.Length];
        float[] next32 = new float[values32.Length], previous32 = new float[values32.Length];
        int[] exponent64 = new int[values64.Length], exponent32 = new int[values32.Length];
        fixture.Gpu.For(0, values64.Length, i =>
        {
            next64[i] = helper ? Increment(values64[i]) : Math.BitIncrement(values64[i]);
            previous64[i] = Math.BitDecrement(values64[i]);
            exponent64[i] = Math.ILogB(values64[i]);
            next32[i] = helper ? Increment(values32[i]) : MathF.BitIncrement(values32[i]);
            previous32[i] = MathF.BitDecrement(values32[i]);
            exponent32[i] = MathF.ILogB(values32[i]);
        });
        for (int i = 0; i < values64.Length; i++)
        {
            Exact(Math.BitIncrement(values64[i]), next64[i]);
            Exact(Math.BitDecrement(values64[i]), previous64[i]);
            Exact(MathF.BitIncrement(values32[i]), next32[i]);
            Exact(MathF.BitDecrement(values32[i]), previous32[i]);
            Assert.Equal(Math.ILogB(values64[i]), exponent64[i]);
            Assert.Equal(MathF.ILogB(values32[i]), exponent32[i]);
        }
    }

    [Fact]
    public void ScalingRoundsOnceAcrossSubnormalBoundariesAndExtremeIntegerExponents()
    {
        double[] seeds64 = [-0.0, 0.0, double.Epsilon, -double.Epsilon, double.MaxValue,
            double.MinValue, 1, -1, 1.5, 3, double.NaN, double.PositiveInfinity,
            BitConverter.Int64BitsToDouble(0x0010000000000000), BitConverter.Int64BitsToDouble(0x000fffffffffffff)];
        float[] seeds32 = [-0.0f, 0.0f, float.Epsilon, -float.Epsilon, float.MaxValue,
            float.MinValue, 1, -1, 1.5f, 3, float.NaN, float.PositiveInfinity,
            BitConverter.Int32BitsToSingle(0x00800000), BitConverter.Int32BitsToSingle(0x007fffff)];
        int[] powers = [int.MinValue, -2200, -1075, -1074, -1023, -1022, -150, -149,
            -127, -126, -54, -25, -2, -1, 0, 1, 2, 24, 53, 127, 128, 1023, 1024, 2200, int.MaxValue];
        int count = seeds64.Length * powers.Length;
        double[] input64 = new double[count], output64 = new double[count];
        float[] input32 = new float[count], output32 = new float[count];
        int[] exponents = new int[count];
        for (int i = 0; i < count; i++)
        {
            input64[i] = seeds64[i / powers.Length];
            input32[i] = seeds32[i / powers.Length];
            exponents[i] = powers[i % powers.Length];
        }
        fixture.Gpu.For(0, count, i =>
        {
            output64[i] = Math.ScaleB(input64[i], exponents[i]);
            output32[i] = MathF.ScaleB(input32[i], exponents[i]);
        });
        for (int i = 0; i < count; i++)
        {
            Exact(Math.ScaleB(input64[i], exponents[i]), output64[i]);
            Exact(MathF.ScaleB(input32[i], exponents[i]), output32[i]);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FusedMultiplyAddUsesOneRoundingEvenWhenTheSeparateProductOverflowsOrCancels(bool helper)
    {
        (double A, double B, double C)[] fixed64 =
        [
            (Math.BitIncrement(1), Math.BitDecrement(1), -1),
            (double.MaxValue, 2, -double.MaxValue), (double.MinValue, 2, double.MaxValue),
            (double.Epsilon, 0.5, double.Epsilon), (-double.Epsilon, 0.5, -0.0),
            (double.Epsilon, 0.5, 0), (-0.0, 1, -0.0), (-0.0, 1, 0),
            (1, 1, -1), (double.PositiveInfinity, 0, 1), (double.PositiveInfinity, 1, double.NegativeInfinity),
            (double.PositiveInfinity, -1, 2), (2, 3, double.PositiveInfinity),
            (double.NaN, 2, 3), (2, double.NaN, 3), (2, 3, double.NaN)
        ];
        (float A, float B, float C)[] fixed32 =
        [
            (MathF.BitIncrement(1), MathF.BitDecrement(1), -1),
            (float.MaxValue, 2, -float.MaxValue), (float.MinValue, 2, float.MaxValue),
            (float.Epsilon, 0.5f, float.Epsilon), (-float.Epsilon, 0.5f, -0.0f),
            (float.Epsilon, 0.5f, 0), (-0.0f, 1, -0.0f), (-0.0f, 1, 0),
            (1, 1, -1), (float.PositiveInfinity, 0, 1), (float.PositiveInfinity, 1, float.NegativeInfinity),
            (float.PositiveInfinity, -1, 2), (2, 3, float.PositiveInfinity),
            (float.NaN, 2, 3), (2, float.NaN, 3), (2, 3, float.NaN)
        ];
        int count = fixed64.Length + 1024;
        double[] a64 = new double[count], b64 = new double[count], c64 = new double[count], result64 = new double[count];
        float[] a32 = new float[count], b32 = new float[count], c32 = new float[count], result32 = new float[count];
        Random random = new(90211);
        byte[] bytes = new byte[8];
        for (int i = 0; i < count; i++)
        {
            if (i < fixed64.Length)
            {
                (a64[i], b64[i], c64[i]) = fixed64[i];
                (a32[i], b32[i], c32[i]) = fixed32[i];
            }
            else
            {
                random.NextBytes(bytes); a64[i] = BitConverter.ToDouble(bytes);
                random.NextBytes(bytes); b64[i] = BitConverter.ToDouble(bytes);
                random.NextBytes(bytes); c64[i] = i % 3 == 0 ? -(a64[i] * b64[i]) : BitConverter.ToDouble(bytes);
                random.NextBytes(bytes); a32[i] = BitConverter.ToSingle(bytes);
                random.NextBytes(bytes); b32[i] = BitConverter.ToSingle(bytes);
                random.NextBytes(bytes); c32[i] = i % 3 == 0 ? -(a32[i] * b32[i]) : BitConverter.ToSingle(bytes);
            }
        }
        fixture.Gpu.For(0, count, i =>
        {
            result64[i] = helper ? Fused(a64[i], b64[i], c64[i]) : Math.FusedMultiplyAdd(a64[i], b64[i], c64[i]);
            result32[i] = helper ? Fused(a32[i], b32[i], c32[i]) : MathF.FusedMultiplyAdd(a32[i], b32[i], c32[i]);
        });
        for (int i = 0; i < count; i++)
        {
            Exact(Math.FusedMultiplyAdd(a64[i], b64[i], c64[i]), result64[i]);
            Exact(MathF.FusedMultiplyAdd(a32[i], b32[i], c32[i]), result32[i]);
        }
        Assert.NotEqual(0.0, result64[0]);
        Assert.NotEqual(0.0f, result32[0]);
        Assert.Equal(double.MaxValue, result64[1]);
        Assert.Equal(float.MaxValue, result32[1]);
    }

    [Fact]
    public void IeeeRemainderUsesAnEvenQuotientAndRetainsSignedZero()
    {
        double[] a64 = [3, 5, 7, 9, -3, -5, -7, -9, 4, -4, 0, -0.0,
            double.MaxValue, double.Epsilon, 7 * double.Epsilon, double.NaN, 1, double.PositiveInfinity, 1, 1];
        double[] b64 = [2, 2, 2, 2, 2, 2, 2, 2, -2, 2, 2, 2,
            double.Epsilon, 2 * double.Epsilon, 2 * double.Epsilon, 1, double.NaN, 1, 0, double.PositiveInfinity];
        float[] a32 = [3, 5, 7, 9, -3, -5, -7, -9, 4, -4, 0, -0.0f,
            float.MaxValue, float.Epsilon, 7 * float.Epsilon, float.NaN, 1, float.PositiveInfinity, 1, 1];
        float[] b32 = [2, 2, 2, 2, 2, 2, 2, 2, -2, 2, 2, 2,
            float.Epsilon, 2 * float.Epsilon, 2 * float.Epsilon, 1, float.NaN, 1, 0, float.PositiveInfinity];
        double[] output64 = new double[a64.Length];
        float[] output32 = new float[a32.Length];
        fixture.Gpu.For(0, a64.Length, i =>
        {
            output64[i] = Math.IEEERemainder(a64[i], b64[i]);
            output32[i] = MathF.IEEERemainder(a32[i], b32[i]);
        });
        for (int i = 0; i < a64.Length; i++)
        {
            Exact(Math.IEEERemainder(a64[i], b64[i]), output64[i]);
            Exact(MathF.IEEERemainder(a32[i], b32[i]), output32[i]);
        }
    }

    [Fact]
    public void MagnitudeExtremaPropagateNanAndChooseTheSignedTie()
    {
        double[] left64 = [0.0, -0.0, 2, -2, double.NaN, 1, double.PositiveInfinity, double.Epsilon];
        double[] right64 = [-0.0, 0.0, -2, 2, 1, double.NaN, double.NegativeInfinity, -double.Epsilon];
        float[] left32 = [0.0f, -0.0f, 2, -2, float.NaN, 1, float.PositiveInfinity, float.Epsilon];
        float[] right32 = [-0.0f, 0.0f, -2, 2, 1, float.NaN, float.NegativeInfinity, -float.Epsilon];
        double[] maximum64 = new double[8], minimum64 = new double[8];
        float[] maximum32 = new float[8], minimum32 = new float[8];
        fixture.Gpu.For(0, 8, i =>
        {
            maximum64[i] = Math.MaxMagnitude(left64[i], right64[i]);
            minimum64[i] = Math.MinMagnitude(left64[i], right64[i]);
            maximum32[i] = MathF.MaxMagnitude(left32[i], right32[i]);
            minimum32[i] = MathF.MinMagnitude(left32[i], right32[i]);
        });
        for (int i = 0; i < 8; i++)
        {
            Exact(Math.MaxMagnitude(left64[i], right64[i]), maximum64[i]);
            Exact(Math.MinMagnitude(left64[i], right64[i]), minimum64[i]);
            Exact(MathF.MaxMagnitude(left32[i], right32[i]), maximum32[i]);
            Exact(MathF.MinMagnitude(left32[i], right32[i]), minimum32[i]);
        }
    }

    [Fact]
    public void ReciprocalEstimatesHandleExceptionalSignsAndFiniteMagnitudes()
    {
        double[] values64 = [double.NaN, double.NegativeInfinity, double.PositiveInfinity, -0.0, 0.0,
            -1, 1, 2, 3, 0.25, 100, double.Epsilon, double.MaxValue];
        float[] values32 = [float.NaN, float.NegativeInfinity, float.PositiveInfinity, -0.0f, 0.0f,
            -1, 1, 2, 3, 0.25f, 100, float.Epsilon, float.MaxValue];
        double[] reciprocal64 = new double[values64.Length], root64 = new double[values64.Length];
        float[] reciprocal32 = new float[values32.Length], root32 = new float[values32.Length];
        fixture.Gpu.For(0, values64.Length, i =>
        {
            reciprocal64[i] = Math.ReciprocalEstimate(values64[i]);
            root64[i] = Math.ReciprocalSqrtEstimate(values64[i]);
            reciprocal32[i] = MathF.ReciprocalEstimate(values32[i]);
            root32[i] = MathF.ReciprocalSqrtEstimate(values32[i]);
        });
        for (int i = 0; i < values64.Length; i++)
        {
            Approximate(1.0 / values64[i], reciprocal64[i], 1e-14);
            Approximate(1.0 / Math.Sqrt(values64[i]), root64[i], 1e-14);
            Approximate((float)(1.0 / values32[i]), reciprocal32[i], 2e-7);
            Approximate((float)(1.0 / Math.Sqrt(values32[i])), root32[i], 2e-7);
        }
    }

    private static double Increment(double value) => Math.BitIncrement(value);
    private static float Increment(float value) => MathF.BitIncrement(value);
    private static double Fused(double left, double right, double addend) => Math.FusedMultiplyAdd(left, right, addend);
    private static float Fused(float left, float right, float addend) => MathF.FusedMultiplyAdd(left, right, addend);
    private static void Exact(double expected, double actual)
    {
        if (double.IsNaN(expected)) Assert.True(double.IsNaN(actual));
        else Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual));
    }
    private static void Exact(float expected, float actual)
    {
        if (float.IsNaN(expected)) Assert.True(float.IsNaN(actual));
        else Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual));
    }
    private static void Approximate(double expected, double actual, double relativeTolerance)
    {
        if (!double.IsFinite(expected) || expected == 0) Exact(expected, actual);
        else Assert.True(Math.Abs(expected - actual) <= Math.Abs(expected) * relativeTolerance || expected == actual,
            $"Expected {expected:R}, got {actual:R}");
    }
}
