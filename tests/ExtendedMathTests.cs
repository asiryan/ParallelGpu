using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ExtendedMathTests(GpuFixture fixture)
{
    [Fact]
    public void FiniteMathWorksInBodiesNestedHelpersAndValueConstructors()
    {
        double[] values64 = [0.0001, 0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 32, 100, 1000, 1048576];
        float[] values32 = values64.Select(value => (float)value).ToArray();
        double[] exponents64 = [-3, -1, -0.5, 0, 0.5, 1, 2, 3, -3, -1, -0.5, 0, 0.5, 1, 2, 3];
        float[] exponents32 = exponents64.Select(value => (float)value).ToArray();
        double[] bases64 = [0.25, 0.5, 2, 10, 0.25, 0.5, 2, 10, 0.25, 0.5, 2, 10, 0.25, 0.5, 2, 10];
        float[] bases32 = bases64.Select(value => (float)value).ToArray();
        float[] results32 = new float[values32.Length * 15];
        double[] results64 = new double[values64.Length * 15];
        fixture.Gpu.For(0, values32.Length, i =>
        {
            int offset = i * 15;
            results32[offset] = MathF.Sin(values32[i]);
            results32[offset + 1] = MathF.Cos(values32[i]);
            results32[offset + 2] = MathF.Pow(values32[i], exponents32[i]);
            results32[offset + 3] = MathF.Log(values32[i]);
            results32[offset + 4] = MathF.Log(values32[i], bases32[i]);
            results32[offset + 5] = SinSingle(values32[i]);
            results32[offset + 6] = CosSingle(values32[i]);
            results32[offset + 7] = PowSingle(values32[i], exponents32[i]);
            results32[offset + 8] = LogSingle(values32[i]);
            results32[offset + 9] = LogBaseSingle(values32[i], bases32[i]);
            var value32 = new MathValueSingle(values32[i], exponents32[i], bases32[i]);
            results32[offset + 10] = value32.Sin;
            results32[offset + 11] = value32.Cos;
            results32[offset + 12] = value32.Pow;
            results32[offset + 13] = value32.Log;
            results32[offset + 14] = value32.LogBase;
            results64[offset] = Math.Sin(values64[i]);
            results64[offset + 1] = Math.Cos(values64[i]);
            results64[offset + 2] = Math.Pow(values64[i], exponents64[i]);
            results64[offset + 3] = Math.Log(values64[i]);
            results64[offset + 4] = Math.Log(values64[i], bases64[i]);
            results64[offset + 5] = SinDouble(values64[i]);
            results64[offset + 6] = CosDouble(values64[i]);
            results64[offset + 7] = PowDouble(values64[i], exponents64[i]);
            results64[offset + 8] = LogDouble(values64[i]);
            results64[offset + 9] = LogBaseDouble(values64[i], bases64[i]);
            var value64 = new MathValueDouble(values64[i], exponents64[i], bases64[i]);
            results64[offset + 10] = value64.Sin;
            results64[offset + 11] = value64.Cos;
            results64[offset + 12] = value64.Pow;
            results64[offset + 13] = value64.Log;
            results64[offset + 14] = value64.LogBase;
        });
        for (int i = 0; i < values32.Length; i++)
        {
            float[] expected32 = [MathF.Sin(values32[i]), MathF.Cos(values32[i]),
                MathF.Pow(values32[i], exponents32[i]), MathF.Log(values32[i]), MathF.Log(values32[i], bases32[i])];
            double[] expected64 = [Math.Sin(values64[i]), Math.Cos(values64[i]),
                Math.Pow(values64[i], exponents64[i]), Math.Log(values64[i]), Math.Log(values64[i], bases64[i])];
            for (int call = 0; call < 3; call++)
                for (int operation = 0; operation < 5; operation++)
                {
                    string context = $"value {i}, call path {call}, operation {operation}";
                    AssertSingle(expected32[operation], results32[i * 15 + call * 5 + operation], context, operation < 2);
                    AssertDouble(expected64[operation], results64[i * 15 + call * 5 + operation], context, operation < 2);
                }
        }
    }

    [Fact]
    public void TrigonometricFunctionsPreserveSignedZeroAndHandleSpecialAndLargeArguments()
    {
        float[] input32 = [0F, -0F, float.Epsilon, -float.Epsilon,
            BitConverter.Int32BitsToSingle(0x007fffff), BitConverter.Int32BitsToSingle(0x00800000),
            -1F, 1F, MathF.PI, -MathF.PI, 1e6F, -1e6F, float.MaxValue, -float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN];
        double[] input64 = [0D, -0D, double.Epsilon, -double.Epsilon,
            BitConverter.Int64BitsToDouble(0x000fffffffffffff), BitConverter.Int64BitsToDouble(0x0010000000000000),
            -1D, 1D, Math.PI, -Math.PI, 1e6D, -1e6D, double.MaxValue, -double.MaxValue,
            double.PositiveInfinity, double.NegativeInfinity, double.NaN];
        float[] sin32 = new float[input32.Length], cos32 = new float[input32.Length];
        double[] sin64 = new double[input64.Length], cos64 = new double[input64.Length];
        fixture.Gpu.For(0, input32.Length, i =>
        {
            sin32[i] = MathF.Sin(input32[i]);
            cos32[i] = CosSingle(input32[i]);
            sin64[i] = SinDouble(input64[i]);
            cos64[i] = Math.Cos(input64[i]);
        });
        for (int i = 0; i < input32.Length; i++)
        {
            AssertSingle(MathF.Sin(input32[i]), sin32[i], $"sin({input32[i]:R})", Math.Abs(input32[i]) >= 1e-8F);
            AssertSingle(MathF.Cos(input32[i]), cos32[i], $"cos({input32[i]:R})", true);
            AssertDouble(Math.Sin(input64[i]), sin64[i], $"sin({input64[i]:R})", Math.Abs(input64[i]) >= 1e-8D);
            AssertDouble(Math.Cos(input64[i]), cos64[i], $"cos({input64[i]:R})", true);
        }
    }

    [Fact]
    public void PowersMatchClrDomainsInfinitiesSignedZeroAndLargeIntegerParity()
    {
        float[] bases32 = [float.NaN, float.NegativeInfinity, -float.MaxValue, -2F, -1F,
            -float.Epsilon, -0F, 0F, float.Epsilon, 0.25F, 0.5F, 1F, 2F, float.MaxValue, float.PositiveInfinity];
        float[] exponents32 = [float.NaN, float.NegativeInfinity, -16777216F, -16777215F,
            -4F, -3F, -2F, -0.5F, -0F, 0F, 0.5F, 1F, 2F, 3F, 4F, 16777215F, 16777216F, float.PositiveInfinity];
        double[] bases64 = [double.NaN, double.NegativeInfinity, -double.MaxValue, -2D, -1D,
            -double.Epsilon, -0D, 0D, double.Epsilon, 0.25D, 0.5D, 1D, 2D, double.MaxValue, double.PositiveInfinity];
        double[] exponents64 = [double.NaN, double.NegativeInfinity, -9007199254740992D, -9007199254740991D,
            -4D, -3D, -2D, -0.5D, -0D, 0D, 0.5D, 1D, 2D, 3D, 4D, 9007199254740991D, 9007199254740992D, double.PositiveInfinity];
        float[] left32 = bases32.SelectMany(value => exponents32.Select(_ => value)).ToArray();
        float[] right32 = bases32.SelectMany(_ => exponents32).ToArray();
        double[] left64 = bases64.SelectMany(value => exponents64.Select(_ => value)).ToArray();
        double[] right64 = bases64.SelectMany(_ => exponents64).ToArray();
        float[] direct32 = new float[left32.Length], helper32 = new float[left32.Length];
        double[] direct64 = new double[left64.Length], constructor64 = new double[left64.Length];
        fixture.Gpu.For(0, left32.Length, i =>
        {
            direct32[i] = MathF.Pow(left32[i], right32[i]);
            helper32[i] = PowSingle(left32[i], right32[i]);
            direct64[i] = Math.Pow(left64[i], right64[i]);
            constructor64[i] = new PowerValueDouble(left64[i], right64[i]).Value;
        });
        for (int i = 0; i < left32.Length; i++)
        {
            float expected32 = MathF.Pow(left32[i], right32[i]);
            double expected64 = Math.Pow(left64[i], right64[i]);
            AssertSingle(expected32, direct32[i], $"direct pow({left32[i]:R}, {right32[i]:R})");
            AssertSingle(expected32, helper32[i], $"helper pow({left32[i]:R}, {right32[i]:R})");
            AssertDouble(expected64, direct64[i], $"direct pow({left64[i]:R}, {right64[i]:R})");
            AssertDouble(expected64, constructor64[i], $"constructor pow({left64[i]:R}, {right64[i]:R})");
        }
    }

    [Fact]
    public void NaturalAndArbitraryBaseLogarithmsMatchClrSpecialValuesAndSubnormals()
    {
        float[] values32 = [float.NaN, float.NegativeInfinity, -1F, -float.Epsilon, -0F, 0F,
            float.Epsilon, BitConverter.Int32BitsToSingle(0x007fffff), BitConverter.Int32BitsToSingle(0x00800000),
            0.5F, 1F, 2F, float.MaxValue, float.PositiveInfinity];
        double[] values64 = [double.NaN, double.NegativeInfinity, -1D, -double.Epsilon, -0D, 0D,
            double.Epsilon, BitConverter.Int64BitsToDouble(0x000fffffffffffff), BitConverter.Int64BitsToDouble(0x0010000000000000),
            0.5D, 1D, 2D, double.MaxValue, double.PositiveInfinity];
        float[] bases32 = [float.NaN, float.NegativeInfinity, -1F, -0F, 0F, float.Epsilon, 0.5F,
            1F, 2F, float.MaxValue, float.PositiveInfinity];
        double[] bases64 = [double.NaN, double.NegativeInfinity, -1D, -0D, 0D, double.Epsilon, 0.5D,
            1D, 2D, double.MaxValue, double.PositiveInfinity];
        float[] left32 = values32.SelectMany(value => bases32.Select(_ => value)).ToArray();
        float[] right32 = values32.SelectMany(_ => bases32).ToArray();
        double[] left64 = values64.SelectMany(value => bases64.Select(_ => value)).ToArray();
        double[] right64 = values64.SelectMany(_ => bases64).ToArray();
        float[] natural32 = new float[left32.Length], arbitrary32 = new float[left32.Length];
        double[] natural64 = new double[left64.Length], arbitrary64 = new double[left64.Length];
        fixture.Gpu.For(0, left32.Length, i =>
        {
            natural32[i] = MathF.Log(left32[i]);
            arbitrary32[i] = LogBaseSingle(left32[i], right32[i]);
            natural64[i] = LogDouble(left64[i]);
            arbitrary64[i] = Math.Log(left64[i], right64[i]);
        });
        for (int i = 0; i < left32.Length; i++)
        {
            AssertSingle(MathF.Log(left32[i]), natural32[i], $"log({left32[i]:R})");
            AssertSingle(MathF.Log(left32[i], right32[i]), arbitrary32[i], $"log({left32[i]:R}, {right32[i]:R})");
            AssertDouble(Math.Log(left64[i]), natural64[i], $"log({left64[i]:R})");
            AssertDouble(Math.Log(left64[i], right64[i]), arbitrary64[i], $"log({left64[i]:R}, {right64[i]:R})");
        }
    }

    [Fact]
    public void DeterministicCorpusCoversTheFiniteExponentRangeAndNearUnityPowers()
    {
        const int count = 512;
        var random = new Random(193711);
        byte[] bytes = new byte[8];
        float[] angles32 = new float[count], positive32 = new float[count], bases32 = new float[count], exponents32 = new float[count];
        double[] angles64 = new double[count], positive64 = new double[count], bases64 = new double[count], exponents64 = new double[count];
        for (int i = 0; i < count; i++)
        {
            random.NextBytes(bytes);
            int fraction32 = BitConverter.ToInt32(bytes, 0) & 0x007fffff;
            long fraction64 = BitConverter.ToInt64(bytes, 0) & 0x000fffffffffffffL;
            // Single precision visits every normal exponent twice; double precision samples
            // evenly spaced exponents across the complete normal range, plus subnormals.
            int exponent32 = i % 254 + 1;
            int exponent64 = 1 + i * 2045 / (count - 1);
            if (i % 64 == 0)
                exponent32 = exponent64 = 0;
            positive32[i] = BitConverter.Int32BitsToSingle((exponent32 << 23) | (fraction32 | 1));
            positive64[i] = BitConverter.Int64BitsToDouble(((long)exponent64 << 52) | (fraction64 | 1L));
            angles32[i] = i % 2 == 0 ? positive32[i] : -positive32[i];
            angles64[i] = i % 2 == 0 ? positive64[i] : -positive64[i];
            if (i % 4 == 0)
            {
                // Very large exponents amplify errors in log(1 + epsilon).
                bases32[i] = BitConverter.Int32BitsToSingle(i % 8 == 0 ? 0x3f800001 : 0x3f7fffff);
                bases64[i] = BitConverter.Int64BitsToDouble(i % 8 == 0 ? 0x3ff0000000000001L : 0x3fefffffffffffffL);
                exponents32[i] = i % 16 < 8 ? 8388608F : -8388608F;
                exponents64[i] = i % 16 < 8 ? 4503599627370496D : -4503599627370496D;
            }
            else
            {
                bases32[i] = BitConverter.Int32BitsToSingle(((126 + i % 2) << 23) | fraction32);
                bases64[i] = BitConverter.Int64BitsToDouble(((1022L + i % 2) << 52) | fraction64);
                exponents32[i] = (i % 17 - 8) * 0.5F;
                exponents64[i] = (i % 17 - 8) * 0.5D;
            }
        }
        float[] sin32 = new float[count], cos32 = new float[count], log32 = new float[count], pow32 = new float[count];
        double[] sin64 = new double[count], cos64 = new double[count], log64 = new double[count], pow64 = new double[count];
        fixture.Gpu.For(0, count, i =>
        {
            sin32[i] = MathF.Sin(angles32[i]);
            cos32[i] = MathF.Cos(angles32[i]);
            log32[i] = MathF.Log(positive32[i]);
            pow32[i] = MathF.Pow(bases32[i], exponents32[i]);
            sin64[i] = Math.Sin(angles64[i]);
            cos64[i] = Math.Cos(angles64[i]);
            log64[i] = Math.Log(positive64[i]);
            pow64[i] = Math.Pow(bases64[i], exponents64[i]);
        });
        for (int i = 0; i < count; i++)
        {
            AssertSingle(MathF.Sin(angles32[i]), sin32[i], $"corpus sin({angles32[i]:R})", Math.Abs(angles32[i]) >= 1e-8F);
            AssertSingle(MathF.Cos(angles32[i]), cos32[i], $"corpus cos({angles32[i]:R})", true);
            AssertSingle(MathF.Log(positive32[i]), log32[i], $"corpus log({positive32[i]:R})");
            AssertSingle(MathF.Pow(bases32[i], exponents32[i]), pow32[i], $"corpus pow({bases32[i]:R}, {exponents32[i]:R})");
            AssertDouble(Math.Sin(angles64[i]), sin64[i], $"corpus sin({angles64[i]:R})", Math.Abs(angles64[i]) >= 1e-8D);
            AssertDouble(Math.Cos(angles64[i]), cos64[i], $"corpus cos({angles64[i]:R})", true);
            AssertDouble(Math.Log(positive64[i]), log64[i], $"corpus log({positive64[i]:R})");
            AssertDouble(Math.Pow(bases64[i], exponents64[i]), pow64[i], $"corpus pow({bases64[i]:R}, {exponents64[i]:R})");
        }
    }

    [Fact]
    public void PowersOfTwoPreserveOverflowNormalAndSubnormalBoundaries()
    {
        float[] bases32 = [2F, 2F, 2F, 2F, 2F, 2F, 0.5F, 0.5F, 0.5F, 4F, -2F, -2F];
        float[] powers32 = [127F, 128F, -126F, -148F, -149F, -150F, -128F, 149F, 150F, 0.5F, -149F, -151F];
        double[] bases64 = [2D, 2D, 2D, 2D, 2D, 2D, 0.5D, 0.5D, 0.5D, 4D, -2D, -2D];
        double[] powers64 = [1023D, 1024D, -1022D, -1073D, -1074D, -1075D, -1024D, 1074D, 1075D, 0.5D, -1073D, -1075D];
        float[] results32 = new float[bases32.Length];
        double[] results64 = new double[bases64.Length];
        fixture.Gpu.For(0, results32.Length, i =>
        {
            results32[i] = MathF.Pow(bases32[i], powers32[i]);
            results64[i] = PowDouble(bases64[i], powers64[i]);
        });
        for (int i = 0; i < results32.Length; i++)
        {
            float expected32 = MathF.Pow(bases32[i], powers32[i]);
            double expected64 = Math.Pow(bases64[i], powers64[i]);
            Assert.True(BitConverter.SingleToInt32Bits(expected32) == BitConverter.SingleToInt32Bits(results32[i]),
                $"power-of-two pow({bases32[i]:R}, {powers32[i]:R}): expected {expected32:R}, actual {results32[i]:R}");
            bool halfwayToFirstSubnormal =
                (Math.Abs(bases64[i]) == 2D && powers64[i] == -1075D) ||
                (Math.Abs(bases64[i]) == 0.5D && powers64[i] == 1075D);
            if (halfwayToFirstSubnormal)
            {
                // The exact result is halfway between zero and Double.Epsilon. Windows
                // Math.Pow rounds this approximation upward; nearest-even gives zero.
                Assert.True(double.IsFinite(results64[i]) && Math.Abs(results64[i] - expected64) <= double.Epsilon &&
                    ((BitConverter.DoubleToInt64Bits(results64[i]) ^ BitConverter.DoubleToInt64Bits(expected64)) & long.MinValue) == 0,
                    $"halfway pow({bases64[i]:R}, {powers64[i]:R}): expected within one subnormal ULP of {expected64:R}, actual {results64[i]:R}");
            }
            else
                Assert.True(BitConverter.DoubleToInt64Bits(expected64) == BitConverter.DoubleToInt64Bits(results64[i]),
                    $"power-of-two pow({bases64[i]:R}, {powers64[i]:R}): expected {expected64:R}, actual {results64[i]:R}");
        }
        double[] neighboringPowers = [Math.BitDecrement(-1075D), Math.BitIncrement(-1075D)];
        double[] neighboringResults = new double[neighboringPowers.Length];
        fixture.Gpu.For(0, neighboringResults.Length, i => neighboringResults[i] = Math.Pow(2D, neighboringPowers[i]));
        Assert.Equal(BitConverter.DoubleToInt64Bits(0D), BitConverter.DoubleToInt64Bits(neighboringResults[0]));
        Assert.Equal(BitConverter.DoubleToInt64Bits(double.Epsilon), BitConverter.DoubleToInt64Bits(neighboringResults[1]));
    }

    private static float SinSingle(float value) => SinSingleInner(value);
    private static float SinSingleInner(float value) => MathF.Sin(value);
    private static float CosSingle(float value) => CosSingleInner(value);
    private static float CosSingleInner(float value) => MathF.Cos(value);
    private static float PowSingle(float value, float exponent) => PowSingleInner(value, exponent);
    private static float PowSingleInner(float value, float exponent) => MathF.Pow(value, exponent);
    private static float LogSingle(float value) => LogSingleInner(value);
    private static float LogSingleInner(float value) => MathF.Log(value);
    private static float LogBaseSingle(float value, float newBase) => LogBaseSingleInner(value, newBase);
    private static float LogBaseSingleInner(float value, float newBase) => MathF.Log(value, newBase);
    private static double SinDouble(double value) => SinDoubleInner(value);
    private static double SinDoubleInner(double value) => Math.Sin(value);
    private static double CosDouble(double value) => CosDoubleInner(value);
    private static double CosDoubleInner(double value) => Math.Cos(value);
    private static double PowDouble(double value, double exponent) => PowDoubleInner(value, exponent);
    private static double PowDoubleInner(double value, double exponent) => Math.Pow(value, exponent);
    private static double LogDouble(double value) => LogDoubleInner(value);
    private static double LogDoubleInner(double value) => Math.Log(value);
    private static double LogBaseDouble(double value, double newBase) => LogBaseDoubleInner(value, newBase);
    private static double LogBaseDoubleInner(double value, double newBase) => Math.Log(value, newBase);

    private readonly struct MathValueSingle(float value, float exponent, float newBase)
    {
        public readonly float Sin = SinSingle(value);
        public readonly float Cos = CosSingle(value);
        public readonly float Pow = PowSingle(value, exponent);
        public readonly float Log = LogSingle(value);
        public readonly float LogBase = LogBaseSingle(value, newBase);
    }

    private readonly struct MathValueDouble(double value, double exponent, double newBase)
    {
        public readonly double Sin = SinDouble(value);
        public readonly double Cos = CosDouble(value);
        public readonly double Pow = PowDouble(value, exponent);
        public readonly double Log = LogDouble(value);
        public readonly double LogBase = LogBaseDouble(value, newBase);
    }

    private readonly struct PowerValueDouble(double value, double exponent)
    {
        public readonly double Value = PowDouble(value, exponent);
    }

    private static void AssertSingle(float expected, float actual, string context, bool trigonometric = false)
    {
        if (float.IsNaN(expected))
            Assert.True(float.IsNaN(actual), $"{context}: expected NaN, actual {actual:R}");
        else if (float.IsInfinity(expected) || expected == 0F)
            Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
                $"{context}: expected {expected:R} ({BitConverter.SingleToInt32Bits(expected):X8}), actual {actual:R} ({BitConverter.SingleToInt32Bits(actual):X8})");
        else
        {
            double tolerance = Math.Max(Math.Abs((double)expected) * 4e-6, trigonometric ? 2e-7 : 4D * float.Epsilon);
            bool canRoundToZero = trigonometric && Math.Abs(expected) >= BitConverter.Int32BitsToSingle(0x00800000);
            Assert.True(float.IsFinite(actual) && (actual != 0F || canRoundToZero) && Math.Abs((double)actual - expected) <= tolerance,
                $"{context}: expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}");
        }
    }

    private static void AssertDouble(double expected, double actual, string context, bool trigonometric = false)
    {
        if (double.IsNaN(expected))
            Assert.True(double.IsNaN(actual), $"{context}: expected NaN, actual {actual:R}");
        else if (double.IsInfinity(expected) || expected == 0D)
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual),
                $"{context}: expected {expected:R} ({BitConverter.DoubleToInt64Bits(expected):X16}), actual {actual:R} ({BitConverter.DoubleToInt64Bits(actual):X16})");
        else
        {
            double tolerance = Math.Max(Math.Abs(expected) * 4e-12, trigonometric ? 2e-14 : 4D * double.Epsilon);
            bool canRoundToZero = trigonometric && Math.Abs(expected) >= BitConverter.Int64BitsToDouble(0x0010000000000000);
            Assert.True(double.IsFinite(actual) && (actual != 0D || canRoundToZero) && Math.Abs(actual - expected) <= tolerance,
                $"{context}: expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}");
        }
    }
}
