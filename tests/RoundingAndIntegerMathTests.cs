using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class RoundingAndIntegerMathTests(GpuFixture fixture)
{
    [Fact]
    public void RoundingOverloadsAndTruncateMatchClrIncludingSignedZeros()
    {
        double[] values64 = [-0.0, 0.0, -double.Epsilon, double.Epsilon, -0.25, 0.25,
            -0.5, 0.5, -1.5, 1.5, -2.5, 2.5, -3.5, 3.5, -2.675, 2.675,
            Math.BitDecrement(0.5), Math.BitIncrement(0.5), Math.BitDecrement(2.5), Math.BitIncrement(2.5),
            -1234.56789, 1234.56789, 8_388_609, 4_503_599_627_370_497, 1e16,
            double.MinValue, double.MaxValue, double.NegativeInfinity, double.PositiveInfinity,
            BitConverter.Int64BitsToDouble(0x7ff8123456789abc)];
        float[] values32 = values64.Select(value => (float)value).ToArray();
        values32[2] = -float.Epsilon;
        values32[3] = float.Epsilon;
        values32[16] = MathF.BitDecrement(0.5f);
        values32[17] = MathF.BitIncrement(0.5f);
        values32[18] = MathF.BitDecrement(2.5f);
        values32[19] = MathF.BitIncrement(2.5f);
        values32[^1] = BitConverter.Int32BitsToSingle(0x7fc12345);
        double[] actual64 = new double[values64.Length * 5];
        float[] actual32 = new float[values32.Length * 5];

        foreach (int digits in new[] { 0, 1, 2, 6, 15 })
        {
            int singleDigits = Math.Min(digits, 6);
            foreach (MidpointRounding mode in Enum.GetValues<MidpointRounding>())
            {
                fixture.Gpu.For(0, values64.Length, i =>
                {
                    int offset = i * 5;
                    actual64[offset] = Math.Round(values64[i]);
                    actual64[offset + 1] = Math.Round(values64[i], digits);
                    actual64[offset + 2] = Math.Round(values64[i], mode);
                    actual64[offset + 3] = Math.Round(values64[i], digits, mode);
                    actual64[offset + 4] = Math.Truncate(values64[i]);
                    actual32[offset] = MathF.Round(values32[i]);
                    actual32[offset + 1] = MathF.Round(values32[i], singleDigits);
                    actual32[offset + 2] = MathF.Round(values32[i], mode);
                    actual32[offset + 3] = MathF.Round(values32[i], singleDigits, mode);
                    actual32[offset + 4] = MathF.Truncate(values32[i]);
                });
                for (int i = 0; i < values64.Length; i++)
                {
                    double[] expected64 = [Math.Round(values64[i]), Math.Round(values64[i], digits),
                        Math.Round(values64[i], mode), Math.Round(values64[i], digits, mode), Math.Truncate(values64[i])];
                    float[] expected32 = [MathF.Round(values32[i]), MathF.Round(values32[i], singleDigits),
                        MathF.Round(values32[i], mode), MathF.Round(values32[i], singleDigits, mode), MathF.Truncate(values32[i])];
                    for (int operation = 0; operation < 5; operation++)
                    {
                        AssertDoubleBits(expected64[operation], actual64[i * 5 + operation]);
                        AssertSingleBits(expected32[operation], actual32[i * 5 + operation]);
                    }
                }
            }
        }
    }

    [Fact]
    public void RoundingAcrossFiniteBitPatternsWorksThroughHelpers()
    {
        const int count = 513;
        var random = new Random(91821);
        byte[] bytes = new byte[8];
        double[] values64 = new double[count];
        float[] values32 = new float[count];
        int[] digits64 = new int[count], digits32 = new int[count];
        MidpointRounding[] modes = new MidpointRounding[count];
        for (int i = 0; i < count; i++)
        {
            do { random.NextBytes(bytes); values64[i] = BitConverter.ToDouble(bytes); } while (!double.IsFinite(values64[i]));
            do { random.NextBytes(bytes); values32[i] = BitConverter.ToSingle(bytes); } while (!float.IsFinite(values32[i]));
            digits64[i] = i % 16;
            digits32[i] = i % 7;
            modes[i] = (MidpointRounding)(i % 5);
        }
        double[] actual64 = new double[count];
        float[] actual32 = new float[count];
        fixture.Gpu.For(0, count, i =>
        {
            actual64[i] = RoundThroughHelper(values64[i], digits64[i], modes[i]);
            actual32[i] = RoundThroughHelper(values32[i], digits32[i], modes[i]);
        });
        for (int i = 0; i < count; i++)
        {
            AssertDoubleBits(Math.Round(values64[i], digits64[i], modes[i]), actual64[i]);
            AssertSingleBits(MathF.Round(values32[i], digits32[i], modes[i]), actual32[i]);
        }
    }

    [Theory]
    [InlineData(0, "digits")]
    [InlineData(1, "digits")]
    [InlineData(2, "digits")]
    [InlineData(3, "digits")]
    [InlineData(4, "mode")]
    [InlineData(5, "mode")]
    [InlineData(6, "digits")]
    public void InvalidRoundingArgumentsDeferAndRollBack(int operation, string parameter)
    {
        double[] values = [double.NaN];
        int[] markers = [41];
        Action<int> body = i =>
        {
            markers[i] = 99;
            switch (operation)
            {
                case 0: values[i] = Math.Round(values[i], -1); break;
                case 1: values[i] = Math.Round(values[i], 16); break;
                case 2: values[i] = MathF.Round((float)values[i], -1); break;
                case 3: values[i] = MathF.Round((float)values[i], 7); break;
                case 4: values[i] = Math.Round(values[i], (MidpointRounding)5); break;
                case 5: values[i] = MathF.Round((float)values[i], (MidpointRounding)(-1)); break;
                case 6: values[i] = Math.Round(values[i], -1, (MidpointRounding)5); break;
            }
        };
        ArgumentException error = operation is 4 or 5
            ? Assert.Throws<ArgumentException>(() => fixture.Gpu.For(0, 1, body))
            : Assert.Throws<ArgumentOutOfRangeException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(parameter, error.ParamName);
        Assert.Equal(41, markers[0]);
        Assert.True(double.IsNaN(values[0]));
        fixture.Gpu.For(0, 1, i => markers[i] = 7);
        Assert.Equal(7, markers[0]);
    }

    [Fact]
    public void SignsMatchEverySupportedSignedTypeAndExceptionalValues()
    {
        double[] values64 = [double.NegativeInfinity, -double.Epsilon, -0.0, 0.0, double.Epsilon, double.PositiveInfinity];
        float[] values32 = [float.NegativeInfinity, -float.Epsilon, -0.0f, 0.0f, float.Epsilon, float.PositiveInfinity];
        sbyte[] values8 = [sbyte.MinValue, -1, 0, 0, 1, sbyte.MaxValue];
        short[] values16 = [short.MinValue, -1, 0, 0, 1, short.MaxValue];
        int[] integers32 = [int.MinValue, -1, 0, 0, 1, int.MaxValue];
        long[] integers64 = [long.MinValue, -1, 0, 0, 1, long.MaxValue];
        int[] actual = new int[values64.Length * 7];
        fixture.Gpu.For(0, values64.Length, i =>
        {
            actual[i * 7] = Math.Sign(values64[i]);
            actual[i * 7 + 1] = Math.Sign(values32[i]);
            actual[i * 7 + 2] = MathF.Sign(values32[i]);
            actual[i * 7 + 3] = Math.Sign(values8[i]);
            actual[i * 7 + 4] = Math.Sign(values16[i]);
            actual[i * 7 + 5] = Math.Sign(integers32[i]);
            actual[i * 7 + 6] = Math.Sign(integers64[i]);
        });
        for (int i = 0; i < values64.Length; i++)
            Assert.All(actual.Skip(i * 7).Take(7), value => Assert.Equal(Math.Sign(values64[i]), value));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void NaNSignDefersArithmeticExceptionAndRollsBack(int operation)
    {
        double[] values = [double.NaN];
        int[] result = [41];
        Assert.Throws<ArithmeticException>(() => fixture.Gpu.For(0, 1, i =>
        {
            result[i] = 99;
            result[i] = operation == 0 ? Math.Sign(values[i]) : operation == 1
                ? Math.Sign((float)values[i]) : MathF.Sign((float)values[i]);
        }));
        Assert.Equal(41, result[0]);
    }

    [Fact]
    public void BigMulPreservesBothProductHalvesAndOutElementReferences()
    {
        const int count = 257;
        long[] left = new long[count], right = new long[count];
        long[] boundaries = [0, 1, -1, long.MinValue, long.MaxValue, int.MinValue, int.MaxValue, 0x123456789abcdef];
        var random = new Random(81283);
        byte[] bytes = new byte[8];
        for (int i = 0; i < count; i++)
        {
            random.NextBytes(bytes);
            left[i] = i < boundaries.Length ? boundaries[i] : BitConverter.ToInt64(bytes);
            random.NextBytes(bytes);
            right[i] = i < boundaries.Length ? boundaries[boundaries.Length - i - 1] : BitConverter.ToInt64(bytes);
        }
        int[] left32 = left.Select(value => (int)value).ToArray(), right32 = right.Select(value => (int)value).ToArray();
        ulong[] unsignedLeft = left.Select(value => (ulong)value).ToArray(), unsignedRight = right.Select(value => (ulong)value).ToArray();
        long[] full32 = new long[count], high64 = new long[count], low64 = new long[count];
        ulong[] highUnsigned = new ulong[count], lowUnsigned = new ulong[count];
        fixture.Gpu.For(0, count, i =>
        {
            full32[i] = Math.BigMul(left32[i], right32[i]);
            high64[i] = Math.BigMul(left[i], right[i], out low64[i]);
            highUnsigned[i] = Math.BigMul(unsignedLeft[i], unsignedRight[i], out lowUnsigned[i]);
        });
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(Math.BigMul(left32[i], right32[i]), full32[i]);
            Assert.Equal(Math.BigMul(left[i], right[i], out long low), high64[i]);
            Assert.Equal(low, low64[i]);
            Assert.Equal(Math.BigMul(unsignedLeft[i], unsignedRight[i], out ulong unsignedLow), highUnsigned[i]);
            Assert.Equal(unsignedLow, lowUnsigned[i]);
        }
    }

    [Fact]
    public void DivRemTupleAndOutOverloadsMatchClrAtEveryIntegerWidth()
    {
        sbyte[] left8 = [sbyte.MinValue, sbyte.MaxValue, -17, 17, 0, 1, -1, 42], right8 = [-1, 2, 5, -5, 3, -1, -1, 7];
        byte[] unsigned8 = [byte.MaxValue, 128, 17, 0, 1, 2, 3, 42], divisor8 = [3, byte.MaxValue, 5, 3, 2, 1, 2, 7];
        short[] left16 = [short.MinValue, short.MaxValue, -17, 17, 0, 1, -1, 42], right16 = [-1, 2, 5, -5, 3, -1, -1, 7];
        ushort[] unsigned16 = [ushort.MaxValue, 32768, 17, 0, 1, 2, 3, 42], divisor16 = [3, ushort.MaxValue, 5, 3, 2, 1, 2, 7];
        int[] left32 = [int.MinValue, int.MaxValue, -17, 17, 0, 1, -1, 42], right32 = [1, 2, 5, -5, 3, -1, -1, 7];
        uint[] unsigned32 = [uint.MaxValue, 0x80000000, 17, 0, 1, 2, 3, 42], divisor32 = [3, uint.MaxValue, 5, 3, 2, 1, 2, 7];
        long[] left64 = [long.MinValue, long.MaxValue, -17, 17, 0, 1, -1, 42], right64 = [1, 2, 5, -5, 3, -1, -1, 7];
        ulong[] unsigned64 = [ulong.MaxValue, 0x8000000000000000, 17, 0, 1, 2, 3, 42], divisor64 = [3, ulong.MaxValue, 5, 3, 2, 1, 2, 7];
        (sbyte, sbyte)[] result8 = new (sbyte, sbyte)[8];
        (byte, byte)[] resultUnsigned8 = new (byte, byte)[8];
        (short, short)[] result16 = new (short, short)[8];
        (ushort, ushort)[] resultUnsigned16 = new (ushort, ushort)[8];
        (int, int)[] result32 = new (int, int)[8];
        (uint, uint)[] resultUnsigned32 = new (uint, uint)[8];
        (long, long)[] result64 = new (long, long)[8];
        (ulong, ulong)[] resultUnsigned64 = new (ulong, ulong)[8];
        int[] quotient32 = new int[8], remainder32 = new int[8];
        long[] quotient64 = new long[8], remainder64 = new long[8];
        fixture.Gpu.For(0, 8, i =>
        {
            result8[i] = Math.DivRem(left8[i], right8[i]);
            resultUnsigned8[i] = Math.DivRem(unsigned8[i], divisor8[i]);
            result16[i] = Math.DivRem(left16[i], right16[i]);
            resultUnsigned16[i] = Math.DivRem(unsigned16[i], divisor16[i]);
            result32[i] = Math.DivRem(left32[i], right32[i]);
            resultUnsigned32[i] = Math.DivRem(unsigned32[i], divisor32[i]);
            result64[i] = Math.DivRem(left64[i], right64[i]);
            resultUnsigned64[i] = Math.DivRem(unsigned64[i], divisor64[i]);
            quotient32[i] = Math.DivRem(left32[i], right32[i], out remainder32[i]);
            quotient64[i] = DivideThroughHelper(left64[i], right64[i], out remainder64[i]);
        });
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(Math.DivRem(left8[i], right8[i]), result8[i]);
            Assert.Equal(Math.DivRem(unsigned8[i], divisor8[i]), resultUnsigned8[i]);
            Assert.Equal(Math.DivRem(left16[i], right16[i]), result16[i]);
            Assert.Equal(Math.DivRem(unsigned16[i], divisor16[i]), resultUnsigned16[i]);
            Assert.Equal(Math.DivRem(left32[i], right32[i]), result32[i]);
            Assert.Equal(Math.DivRem(unsigned32[i], divisor32[i]), resultUnsigned32[i]);
            Assert.Equal(Math.DivRem(left64[i], right64[i]), result64[i]);
            Assert.Equal(Math.DivRem(unsigned64[i], divisor64[i]), resultUnsigned64[i]);
            Assert.Equal(result32[i], (quotient32[i], remainder32[i]));
            Assert.Equal(result64[i], (quotient64[i], remainder64[i]));
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    public void DivRemZeroDivisorsDeferAndRollBackAllWrites(int operation)
    {
        int[] zero = [0], result = [41];
        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, i =>
        {
            result[i] = 99;
            switch (operation)
            {
                case 0: result[i] = Math.DivRem(7, zero[i], out result[i]); break;
                case 1: result[i] = (int)Math.DivRem(7L, zero[i], out long remainder); break;
                case 2: result[i] = Math.DivRem((sbyte)7, (sbyte)zero[i]).Quotient; break;
                case 3: result[i] = Math.DivRem((byte)7, (byte)zero[i]).Quotient; break;
                case 4: result[i] = Math.DivRem((short)7, (short)zero[i]).Quotient; break;
                case 5: result[i] = Math.DivRem((ushort)7, (ushort)zero[i]).Quotient; break;
                case 6: result[i] = Math.DivRem(7, zero[i]).Quotient; break;
                case 7: result[i] = (int)Math.DivRem(7U, (uint)zero[i]).Quotient; break;
                case 8: result[i] = (int)Math.DivRem(7L, zero[i]).Quotient; break;
                case 9: result[i] = (int)Math.DivRem(7UL, (ulong)zero[i]).Quotient; break;
            }
        }));
        Assert.Equal(41, result[0]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void DivRemSignedOverflowDefersAndRollsBackOutWrites(int operation)
    {
        int[] divisor = [-1], result = [41];
        Assert.Throws<OverflowException>(() => fixture.Gpu.For(0, 1, i =>
        {
            result[i] = 99;
            switch (operation)
            {
                case 0: result[i] = Math.DivRem(int.MinValue, divisor[i], out result[i]); break;
                case 1: result[i] = (int)Math.DivRem(long.MinValue, divisor[i], out long remainder); break;
                case 2: result[i] = Math.DivRem(int.MinValue, divisor[i]).Quotient; break;
                case 3: result[i] = (int)Math.DivRem(long.MinValue, divisor[i]).Quotient; break;
            }
        }));
        Assert.Equal(41, result[0]);
    }

    private static double RoundThroughHelper(double value, int digits, MidpointRounding mode) => Math.Round(value, digits, mode);
    private static float RoundThroughHelper(float value, int digits, MidpointRounding mode) => MathF.Round(value, digits, mode);
    private static long DivideThroughHelper(long left, long right, out long remainder) => Math.DivRem(left, right, out remainder);
    private static void AssertDoubleBits(double expected, double actual) =>
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual));
    private static void AssertSingleBits(float expected, float actual) =>
        Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual));
}
