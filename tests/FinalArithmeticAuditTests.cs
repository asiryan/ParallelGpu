using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class FinalArithmeticAuditTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void SignedIntegerDivisionAndRemainderMatchClrAtBothWidths()
    {
        int[] left32 = [int.MinValue, int.MaxValue, -17, 17, -17, 17, 0];
        int[] right32 = [1, 2, 5, -5, -1, -1, 3];
        int[] quotient32 = new int[left32.Length];
        int[] remainder32 = new int[left32.Length];
        long[] left64 = [long.MinValue, long.MaxValue, -17, 17, -17, 17, 0];
        long[] right64 = [1, 3, 5, -5, -1, -1, 3];
        long[] quotient64 = new long[left64.Length];
        long[] remainder64 = new long[left64.Length];
        Gpu.For(0, left32.Length, i =>
        {
            quotient32[i] = left32[i] / right32[i];
            remainder32[i] = left32[i] % right32[i];
            quotient64[i] = left64[i] / right64[i];
            remainder64[i] = left64[i] % right64[i];
        });
        Assert.Equal(left32.Zip(right32, (left, right) => left / right), quotient32);
        Assert.Equal(left32.Zip(right32, (left, right) => left % right), remainder32);
        Assert.Equal(left64.Zip(right64, (left, right) => left / right), quotient64);
        Assert.Equal(left64.Zip(right64, (left, right) => left % right), remainder64);
    }

    [Fact]
    public void UnsignedIntegerDivisionAndRemainderPreserveHighBitsAtBothWidths()
    {
        uint[] left32 = [uint.MaxValue, 0, 1, 0x80000000, uint.MaxValue];
        uint[] right32 = [3, 1, 2, uint.MaxValue, 0x80000000];
        uint[] quotient32 = new uint[left32.Length];
        uint[] remainder32 = new uint[left32.Length];
        ulong[] left64 = [ulong.MaxValue, 0, 1, 0x8000000000000000, ulong.MaxValue];
        ulong[] right64 = [3, 1, 2, ulong.MaxValue, 0x8000000000000000];
        ulong[] quotient64 = new ulong[left64.Length];
        ulong[] remainder64 = new ulong[left64.Length];
        Gpu.For(0, left32.Length, i =>
        {
            quotient32[i] = left32[i] / right32[i];
            remainder32[i] = left32[i] % right32[i];
            quotient64[i] = left64[i] / right64[i];
            remainder64[i] = left64[i] % right64[i];
        });
        Assert.Equal(left32.Zip(right32, (left, right) => left / right), quotient32);
        Assert.Equal(left32.Zip(right32, (left, right) => left % right), remainder32);
        Assert.Equal(left64.Zip(right64, (left, right) => left / right), quotient64);
        Assert.Equal(left64.Zip(right64, (left, right) => left % right), remainder64);
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
    public void ZeroDivisorsDeferBothIntegerOperationsAndRollBackEveryCollection(int operation)
    {
        int[] left32 = [7], zero32 = [0];
        uint[] unsigned32 = [uint.MaxValue], unsignedZero32 = [0];
        long[] left64 = [long.MaxValue], zero64 = [0];
        ulong[] unsigned64 = [ulong.MaxValue], unsignedZero64 = [0];
        long[] output = [41];
        Assert.Throws<DivideByZeroException>(() => Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            left32[i]++;
            switch (operation)
            {
                case 0: output[i] = left32[i] / zero32[i]; break;
                case 1: output[i] = left32[i] % zero32[i]; break;
                case 2: output[i] = unsigned32[i] / unsignedZero32[i]; break;
                case 3: output[i] = unsigned32[i] % unsignedZero32[i]; break;
                case 4: output[i] = left64[i] / zero64[i]; break;
                case 5: output[i] = left64[i] % zero64[i]; break;
                case 6: output[i] = unchecked((long)(unsigned64[i] / unsignedZero64[i])); break;
                case 7: output[i] = unchecked((long)(unsigned64[i] % unsignedZero64[i])); break;
            }
        }));
        Assert.Equal(41, output[0]);
        Assert.Equal(7, left32[0]);
        Recover();
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void SignedMinimumWithMinusOneMatchesActualClrDivisionOrRemainder(bool wide, bool remainder)
    {
        int[] left32 = [int.MinValue], right32 = [-1];
        long[] left64 = [long.MinValue], right64 = [-1];
        long expected = 0;
        var hostFailure = Record.Exception(() => expected = wide
            ? remainder ? left64[0] % right64[0] : left64[0] / right64[0]
            : remainder ? left32[0] % right32[0] : left32[0] / right32[0]);
        long[] output = [41];
        var deviceFailure = Record.Exception(() => Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            output[i] = wide
                ? remainder ? left64[i] % right64[i] : left64[i] / right64[i]
                : remainder ? left32[i] % right32[i] : left32[i] / right32[i];
        }));
        Assert.Equal(hostFailure?.GetType(), deviceFailure?.GetType());
        Assert.Equal(hostFailure is null ? expected : 41, output[0]);
        Recover();
    }

    [Fact]
    public void DivisionInsidePrimitiveHelpersDefersErrorsAndCachedCallsRecover()
    {
        int[] left = [17], right = [0], output = [41];
        Action<int> kernel = i =>
        {
            output[i] = 99;
            output[i] = DivideAndRemainder(left[i], right[i]);
        };
        Assert.Throws<DivideByZeroException>(() => Gpu.For(0, 1, kernel));
        Assert.Equal(41, output[0]);
        right[0] = -5;
        Gpu.For(0, 1, kernel);
        Assert.Equal(DivideAndRemainder(left[0], right[0]), output[0]);
        Recover();
    }

    [Fact]
    public void ClampInsidePrimitiveHelpersPreservesBoundsAndRejectsInvertedRanges()
    {
        int[] input = [-7, 3, 20], minimum = [0], maximum = [10], output = [41, 43, 47];
        Action<int> kernel = i =>
        {
            output[i] = 99;
            output[i] = ClampInteger(input[i], minimum[0], maximum[0]);
        };
        Gpu.For(0, input.Length, kernel);
        Assert.Equal(new[] { 0, 3, 10 }, output);
        minimum[0] = 11;
        Assert.Throws<ArgumentException>(() => Gpu.For(0, input.Length, kernel));
        Assert.Equal(new[] { 0, 3, 10 }, output);
        minimum[0] = -10;
        Gpu.For(0, input.Length, kernel);
        Assert.Equal(new[] { -7, 3, 10 }, output);
        Recover();
    }

    [Fact]
    public void FloatingClampMatchesClrNanPayloadsSignedZeroAndInfiniteBounds()
    {
        float nan32 = BitConverter.Int32BitsToSingle(0x7fc12345);
        double nan64 = BitConverter.Int64BitsToDouble(0x7ff8123456789abc);
        float[] values32 = [nan32, 1, 20, -20, -0.0f, 0.0f, float.PositiveInfinity, float.NegativeInfinity];
        float[] minimum32 = [-10, nan32, nan32, -10, 0, -0.0f, float.NegativeInfinity, float.NegativeInfinity];
        float[] maximum32 = [10, 10, 10, nan32, 1, 1, float.PositiveInfinity, float.PositiveInfinity];
        float[] result32 = new float[values32.Length];
        double[] values64 = [nan64, 1, 20, -20, -0.0, 0.0, double.PositiveInfinity, double.NegativeInfinity];
        double[] minimum64 = [-10, nan64, nan64, -10, 0, -0.0, double.NegativeInfinity, double.NegativeInfinity];
        double[] maximum64 = [10, 10, 10, nan64, 1, 1, double.PositiveInfinity, double.PositiveInfinity];
        double[] result64 = new double[values64.Length];
        Gpu.For(0, values32.Length, i =>
        {
            result32[i] = Math.Clamp(values32[i], minimum32[i], maximum32[i]);
            result64[i] = ClampDouble(values64[i], minimum64[i], maximum64[i]);
        });
        Assert.Equal(values32.Select((value, i) => BitConverter.SingleToInt32Bits(Math.Clamp(value, minimum32[i], maximum32[i]))),
            result32.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(values64.Select((value, i) => BitConverter.DoubleToInt64Bits(Math.Clamp(value, minimum64[i], maximum64[i]))),
            result64.Select(BitConverter.DoubleToInt64Bits));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FloatingClampRejectsInvertedBoundsEvenForNanValuesAndRollsBack(bool wide)
    {
        float[] value32 = [float.NaN], min32 = [2], max32 = [1], result32 = [41];
        double[] value64 = [double.NaN], min64 = [2], max64 = [1], result64 = [43];
        Assert.Throws<ArgumentException>(() => Gpu.For(0, 1, i =>
        {
            result32[i] = 99;
            result64[i] = 99;
            if (wide)
                result64[i] = Math.Clamp(value64[i], min64[i], max64[i]);
            else
                result32[i] = Math.Clamp(value32[i], min32[i], max32[i]);
        }));
        Assert.Equal(41, result32[0]);
        Assert.Equal(43, result64[0]);
        Recover();
    }

    [Fact]
    public void FrameworkMinMaxPreserveNanPayloadsAndSignedZero()
    {
        float nan32a = BitConverter.Int32BitsToSingle(0x7fc12345), nan32b = BitConverter.Int32BitsToSingle(0x7fc6789a);
        double nan64a = BitConverter.Int64BitsToDouble(0x7ff8123456789abc), nan64b = BitConverter.Int64BitsToDouble(0x7ff8abcdef123456);
        float[] left32 = [nan32a, 1, nan32a, -0.0f, 0.0f, -0.0f, 0.0f, float.PositiveInfinity, float.NegativeInfinity];
        float[] right32 = [1, nan32b, nan32b, 0.0f, -0.0f, -0.0f, 0.0f, float.PositiveInfinity, float.NegativeInfinity];
        float[] min32 = new float[left32.Length], max32 = new float[left32.Length];
        float[] mathMin32 = new float[left32.Length], mathMax32 = new float[left32.Length];
        double[] left64 = [nan64a, 1, nan64a, -0.0, 0.0, -0.0, 0.0, double.PositiveInfinity, double.NegativeInfinity];
        double[] right64 = [1, nan64b, nan64b, 0.0, -0.0, -0.0, 0.0, double.PositiveInfinity, double.NegativeInfinity];
        double[] min64 = new double[left64.Length], max64 = new double[left64.Length];
        Gpu.For(0, left32.Length, i =>
        {
            min32[i] = MathF.Min(left32[i], right32[i]);
            max32[i] = MathF.Max(left32[i], right32[i]);
            mathMin32[i] = Math.Min(left32[i], right32[i]);
            mathMax32[i] = Math.Max(left32[i], right32[i]);
            min64[i] = Math.Min(left64[i], right64[i]);
            max64[i] = Math.Max(left64[i], right64[i]);
        });
        Assert.Equal(left32.Zip(right32, (left, right) => BitConverter.SingleToInt32Bits(MathF.Min(left, right))),
            min32.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(left32.Zip(right32, (left, right) => BitConverter.SingleToInt32Bits(MathF.Max(left, right))),
            max32.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(left32.Zip(right32, (left, right) => BitConverter.SingleToInt32Bits(Math.Min(left, right))),
            mathMin32.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(left32.Zip(right32, (left, right) => BitConverter.SingleToInt32Bits(Math.Max(left, right))),
            mathMax32.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(left64.Zip(right64, (left, right) => BitConverter.DoubleToInt64Bits(Math.Min(left, right))),
            min64.Select(BitConverter.DoubleToInt64Bits));
        Assert.Equal(left64.Zip(right64, (left, right) => BitConverter.DoubleToInt64Bits(Math.Max(left, right))),
            max64.Select(BitConverter.DoubleToInt64Bits));
    }

    [Fact]
    public void UnsupportedFrameworkRoundingIsRejectedBeforeCompilationInDirectAndHelperCalls()
    {
        double[] input = [2.5], output = [41];
        Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i => output[i] = Math.Round(input[i])));
        Assert.Equal(41, output[0]);
        Recover();
        Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i => output[i] = RoundDouble(input[i])));
        Assert.Equal(41, output[0]);
        Recover();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void SignedAbsoluteMinimumDefersOverflowForEverySupportedWidth(int width)
    {
        sbyte[] value8 = [sbyte.MinValue];
        short[] value16 = [short.MinValue];
        int[] value32 = [int.MinValue];
        long[] value64 = [long.MinValue];
        long[] output = [41];
        Assert.Throws<OverflowException>(() => Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            switch (width)
            {
                case 0: output[i] = Math.Abs(value8[i]); break;
                case 1: output[i] = Math.Abs(value16[i]); break;
                case 2: output[i] = AbsoluteInteger(value32[i]); break;
                case 3: output[i] = Math.Abs(value64[i]); break;
            }
        }));
        Assert.Equal(41, output[0]);
        Recover();
    }

    [Fact]
    public void NarrowingCastsAndMixedConditionalIntegerOperandsKeepCilDivisionWidths()
    {
        int[] input = [65535, 65535, 32768, 128], divisors = [2, 2, 2, 2];
        uint[] unsigned32 = [uint.MaxValue, 65535, 0x80000000, 127];
        uint[] divisor32 = [3, 3, uint.MaxValue, 2];
        ulong[] unsigned64 = [ulong.MaxValue, 65535, 0x8000000000000000, 127];
        long[] signed64 = [-1, 65535, long.MinValue, 127];
        ulong[] divisor64 = [3, 3, ulong.MaxValue, 2];
        int[] narrow = new int[input.Length];
        uint[] result32 = new uint[input.Length];
        ulong[] result64 = new ulong[input.Length];
        Gpu.For(0, input.Length, i =>
        {
            narrow[i] = (i % 2 == 0 ? unchecked((short)input[i]) : unchecked((byte)input[i])) / unchecked((sbyte)divisors[i]);
            result32[i] = (i % 2 == 0 ? unsigned32[i] : unchecked((ushort)input[i])) / divisor32[i];
            result64[i] = (i % 2 == 0 ? unsigned64[i] : unchecked((ulong)signed64[i])) / divisor64[i];
        });
        Assert.Equal(input.Select((value, i) => (i % 2 == 0 ? unchecked((short)value) : unchecked((byte)value)) / unchecked((sbyte)divisors[i])), narrow);
        Assert.Equal(input.Select((value, i) => (i % 2 == 0 ? unsigned32[i] : unchecked((ushort)value)) / divisor32[i]), result32);
        Assert.Equal(input.Select((_, i) => (i % 2 == 0 ? unsigned64[i] : unchecked((ulong)signed64[i])) / divisor64[i]), result64);
    }

    [Fact]
    public void UnsignedClampUsesUnsignedBoundsAtEverySupportedWidth()
    {
        ushort[] input16 = [1, 32769, ushort.MaxValue], min16 = [32768], max16 = [65000], result16 = new ushort[3];
        uint[] input32 = [1, 0x80000001, uint.MaxValue], min32 = [0x80000000], max32 = [uint.MaxValue - 1], result32 = new uint[3];
        ulong[] input64 = [1, 0x8000000000000001, ulong.MaxValue], min64 = [0x8000000000000000], max64 = [ulong.MaxValue - 1], result64 = new ulong[3];
        Gpu.For(0, input16.Length, i =>
        {
            result16[i] = Math.Clamp(input16[i], min16[0], max16[0]);
            result32[i] = Math.Clamp(input32[i], min32[0], max32[0]);
            result64[i] = Math.Clamp(input64[i], min64[0], max64[0]);
        });
        Assert.Equal(input16.Select(value => Math.Clamp(value, min16[0], max16[0])), result16);
        Assert.Equal(input32.Select(value => Math.Clamp(value, min32[0], max32[0])), result32);
        Assert.Equal(input64.Select(value => Math.Clamp(value, min64[0], max64[0])), result64);
        min64[0] = ulong.MaxValue;
        Assert.Throws<ArgumentException>(() => Gpu.For(0, 1, i =>
        {
            result16[i] = 99;
            result64[i] = Math.Clamp(input64[i], min64[0], max64[0]);
        }));
        Assert.Equal((ushort)32768, result16[0]);
        Assert.Equal(0x8000000000000000, result64[0]);
        Recover();
    }

    [Fact]
    public void AllowedNumericIntrinsicsRetainNormalFloatingAndUnsignedIntegerBehavior()
    {
        double[] input = [0.25, 0.5, 1], root = new double[3];
        float[] floating = [-3, 2, -0.0f], absolute = new float[3], minimum = new float[3], maximum = new float[3];
        uint[] left32 = [uint.MaxValue, 0x80000000, 1], right32 = [0x80000000, uint.MaxValue, 2], min32 = new uint[3], max32 = new uint[3];
        ulong[] left64 = [ulong.MaxValue, 0x8000000000000000, 1], right64 = [0x8000000000000000, ulong.MaxValue, 2], min64 = new ulong[3], max64 = new ulong[3];
        Gpu.For(0, input.Length, i =>
        {
            root[i] = Math.Sqrt(input[i]);
            absolute[i] = MathF.Abs(floating[i]);
            minimum[i] = MathF.Min(floating[i], 1);
            maximum[i] = MathF.Max(floating[i], 1);
            min32[i] = Math.Min(left32[i], right32[i]);
            max32[i] = Math.Max(left32[i], right32[i]);
            min64[i] = Math.Min(left64[i], right64[i]);
            max64[i] = Math.Max(left64[i], right64[i]);
        });
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(Math.Sqrt(input[i]), root[i], 12);
        }
        Assert.Equal(floating.Select(value => BitConverter.SingleToInt32Bits(MathF.Abs(value))), absolute.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(floating.Select(value => MathF.Min(value, 1)), minimum);
        Assert.Equal(floating.Select(value => MathF.Max(value, 1)), maximum);
        Assert.Equal(left32.Zip(right32, Math.Min), min32);
        Assert.Equal(left32.Zip(right32, Math.Max), max32);
        Assert.Equal(left64.Zip(right64, Math.Min), min64);
        Assert.Equal(left64.Zip(right64, Math.Max), max64);
    }

    [Fact]
    public void CheckedAdditionIsRejectedBeforeKernelCompilationAndFollowingLoopsWork()
    {
        int[] input = [int.MaxValue], output = [41];
        var failure = Record.Exception(() => Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            output[i] = checked(input[i] + 1);
        }));
        Assert.Equal(41, output[0]);
        Recover();
        Assert.IsType<NotSupportedException>(failure);
    }

    [Fact]
    public void CheckedSubtractionIsRejectedBeforeKernelCompilationAndFollowingLoopsWork()
    {
        uint[] input = [0], output = [41];
        var failure = Record.Exception(() => Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            output[i] = checked(input[i] - 1u);
        }));
        Assert.Equal(41u, output[0]);
        Recover();
        Assert.IsType<NotSupportedException>(failure);
    }

    [Fact]
    public void CheckedMultiplicationIsRejectedBeforeKernelCompilationAndFollowingLoopsWork()
    {
        long[] input = [long.MaxValue], output = [41];
        var failure = Record.Exception(() => Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            output[i] = checked(input[i] * 2);
        }));
        Assert.Equal(41, output[0]);
        Recover();
        Assert.IsType<NotSupportedException>(failure);
    }

    [Fact]
    public void CheckedNarrowingConversionIsRejectedBeforeKernelCompilationAndFollowingLoopsWork()
    {
        long[] input = [long.MaxValue];
        int[] output = [41];
        var failure = Record.Exception(() => Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            output[i] = checked((int)input[i]);
        }));
        Assert.Equal(41, output[0]);
        Recover();
        Assert.IsType<NotSupportedException>(failure);
    }

    [Fact]
    public void CheckedInstructionsInsideHelpersAreRejectedBeforeCompilationAndFollowingLoopsWork()
    {
        int[] input = [int.MaxValue], output = [41];
        var failure = Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            output[i] = CheckedIncrement(input[i]);
        }));
        Assert.Contains("Checked instruction", failure.Message, StringComparison.Ordinal);
        Assert.Equal(41, output[0]);
        Recover();
    }

    private static int CheckedIncrement(int value) => checked(value + 1);
    [Fact]
    public void FrameworkPowerAndSineExecuteInDirectAndHelperCalls()
    {
        double[] input = [2], power = [41], helperPower = [41], sine = [41], helperSine = [41];
        Gpu.For(0, 1, i =>
        {
            power[i] = Math.Pow(input[i], 2);
            helperPower[i] = PowerDouble(input[i]);
            sine[i] = Math.Sin(input[i]);
            helperSine[i] = SineDouble(input[i]);
        });
        Assert.Equal(4, power[0]);
        Assert.Equal(4, helperPower[0]);
        Assert.Equal(Math.Sin(input[0]), sine[0], 12);
        Assert.Equal(Math.Sin(input[0]), helperSine[0], 12);
        Recover();
    }

    private static double PowerDouble(double value) => Math.Pow(value, 2);
    private static double SineDouble(double value) => Math.Sin(value);
    [Fact]
    public void MixedSmallArrayAndStructureFieldLoadsUseCilIntegerWidthWithoutExplicitCasts()
    {
        short[] signed = [short.MinValue, -1, 32767, 128];
        byte[] unsigned = [255, 255, 128, 128];
        NarrowFields[] fields = signed.Select((value, i) => new NarrowFields { Signed = value, Unsigned = unsigned[i] }).ToArray();
        int[] arrayResult = new int[signed.Length], fieldResult = new int[signed.Length];
        Gpu.For(0, signed.Length, i =>
        {
            arrayResult[i] = (i % 2 == 0 ? signed[i] : unsigned[i]) / 2;
            fieldResult[i] = (i % 2 == 0 ? fields[i].Signed : fields[i].Unsigned) / 2;
        });
        var expected = signed.Select((value, i) => (i % 2 == 0 ? value : unsigned[i]) / 2).ToArray();
        Assert.Equal(expected, arrayResult);
        Assert.Equal(expected, fieldResult);
    }

    public struct NarrowFields
    {
        public short Signed;
        public byte Unsigned;
    }
    private static int AbsoluteInteger(int value) => Math.Abs(value);
    private static int DivideAndRemainder(int left, int right) => left / right + left % right;
    private static int ClampInteger(int value, int minimum, int maximum) => Math.Clamp(value, minimum, maximum);
    private static double ClampDouble(double value, double minimum, double maximum) => Math.Clamp(value, minimum, maximum);
    private static double RoundDouble(double value) => Math.Round(value);

    private void Recover()
    {
        int[] output = [41];
        Gpu.For(0, output.Length, i => output[i] = i + 17);
        Assert.Equal(17, output[0]);
    }
}
