using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ExponentialMathTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StandardSingleExponentialHandlesSpecialValuesAndFiniteRange(bool helper)
    {
        float[] input = [float.NaN, float.NegativeInfinity, float.PositiveInfinity, -0.0f, 0.0f,
            -float.Epsilon, float.Epsilon, -120, -104, -103.9f, -103, -100, -90, -87.3f, -80, -10,
            -1, -0.1f, 0.1f, 1, 10, 30, 80, 88, 88.72283f, 89, 100];
        float[] output = new float[input.Length];
        fixture.Gpu.For(0, input.Length, index => output[index] = helper
            ? ExpSingle(input[index]) : MathF.Exp(input[index]));
        for (int i = 0; i < input.Length; i++)
            AssertSingle(MathF.Exp(input[i]), output[i], $"MathF.Exp({input[i]:R})");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StandardDoubleExponentialHandlesSpecialValuesAndFiniteRange(bool helper)
    {
        double[] input = [double.NaN, double.NegativeInfinity, double.PositiveInfinity, -0.0, 0.0,
            -double.Epsilon, double.Epsilon, -740,
            -720, -710, -709, -708.5, -700, -500, -20, -1, -1e-9, 1e-9, 0.1, 1, 20, 100, 500,
            700, 709, 709.7, 709.7827128933839, 709.782712893384, 709.7827128933841, 711, 1000];
        double[] output = new double[input.Length];
        fixture.Gpu.For(0, input.Length, index => output[index] = helper
            ? ExpDouble(input[index]) : Math.Exp(input[index]));
        for (int i = 0; i < input.Length; i++)
            AssertDouble(Math.Exp(input[i]), output[i], $"Math.Exp({input[i]:R})");
    }

    [Fact]
    public void ExponentialUnderflowPreservesMathematicallyRoundedSubnormalBits()
    {
        double[] input64 = [double.NegativeInfinity, -1000, -746, -745.14, -745.13322,
            -745.13321, -745.13, -745, -744.5, -744, -743, -742];
        long[] expected64 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 4, 11];
        float[] input32 = [float.NegativeInfinity, -120, -104, -103.9f, -103, -102, -101, -100];
        int[] expected32 = [0, 0, 0, 1, 1, 4, 10, 27];
        double[] output64 = new double[input64.Length];
        float[] output32 = new float[input32.Length];
        fixture.Gpu.For(0, input64.Length, index => output64[index] = Math.Exp(input64[index]));
        fixture.Gpu.For(0, input32.Length, index => output32[index] = MathF.Exp(input32[index]));
        // Lowest-edge references use the mathematical rounded bits independently of host FTZ/libm behavior.
        Assert.Equal(expected64, output64.Select(BitConverter.DoubleToInt64Bits));
        Assert.Equal(expected32, output32.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void StandardExponentialWorksInValueConstructorsAndRefreshesCachedCaptures()
    {
        float shift32 = 0.5f;
        double shift64 = 0.5;
        float[] output32 = [0];
        double[] output64 = [0];
        Action<int> body = index =>
        {
            var values = new Exponentials(shift32, shift64);
            output32[index] = values.Single;
            output64[index] = values.Double;
        };
        fixture.Gpu.For(0, 1, body);
        AssertSingle(MathF.Exp(shift32), output32[0], "captured single");
        AssertDouble(Math.Exp(shift64), output64[0], "captured double");
        shift32 = float.NaN;
        shift64 = double.NaN;
        fixture.Gpu.For(0, 1, body);
        Assert.True(float.IsNaN(output32[0]));
        Assert.True(double.IsNaN(output64[0]));
        shift32 = -0.5f;
        shift64 = -0.5;
        fixture.Gpu.For(0, 1, body);
        AssertSingle(MathF.Exp(shift32), output32[0], "refreshed single");
        AssertDouble(Math.Exp(shift64), output64[0], "refreshed double");
    }

    [Fact]
    public void StandardLogarithmsPreserveSubnormalInputsAndExceptionalValues()
    {
        float[] input32 = [float.NaN, float.NegativeInfinity, float.PositiveInfinity, -0.0f, 0.0f,
            -1, float.Epsilon, BitConverter.Int32BitsToSingle(0x007fffff), 0.125f, 0.5f, 1, 2, 10];
        double[] input64 = [double.NaN, double.NegativeInfinity, double.PositiveInfinity, -0.0, 0.0,
            -1, double.Epsilon, BitConverter.Int64BitsToDouble(0x000fffffffffffff),
            Math.ScaleB(1.0, -1050), 0.5, 1, 2, 10];
        float[] output32 = new float[input32.Length];
        double[] output64 = new double[input64.Length];
        fixture.Gpu.For(0, input32.Length, index =>
        {
            output32[index] = MathF.Log(input32[index]);
            output64[index] = Math.Log(input64[index]);
        });
        for (int i = 0; i < input32.Length; i++)
        {
            AssertSingle(MathF.Log(input32[i]), output32[i], $"MathF.Log({input32[i]:R})");
            AssertDouble(Math.Log(input64[i]), output64[i], $"Math.Log({input64[i]:R})");
        }
    }

    private static float ExpSingle(float value) => MathF.Exp(value);
    private static double ExpDouble(double value) => ExpDoubleInner(value);
    private static double ExpDoubleInner(double value) => Math.Exp(value);

    private readonly struct Exponentials(float single, double @double)
    {
        public readonly float Single = MathF.Exp(single);
        public readonly double Double = ExpDouble(@double);
    }

    private static void AssertSingle(float expected, float actual, string context)
    {
        if (float.IsNaN(expected))
            Assert.True(float.IsNaN(actual), $"{context}: expected NaN, actual {actual:R}");
        else if (float.IsInfinity(expected) || expected == 0)
            Assert.Equal(expected, actual);
        else
        {
            double tolerance = Math.Max(Math.Abs((double)expected) * 3e-7, 4D * float.Epsilon);
            Assert.True(float.IsFinite(actual) && actual != 0 && Math.Abs((double)expected - actual) <= tolerance,
                $"{context}: expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}");
        }
    }

    private static void AssertDouble(double expected, double actual, string context)
    {
        if (double.IsNaN(expected))
            Assert.True(double.IsNaN(actual), $"{context}: expected NaN, actual {actual:R}");
        else if (double.IsInfinity(expected) || expected == 0)
            Assert.Equal(expected, actual);
        else
        {
            double tolerance = Math.Max(Math.Abs(expected) * 4e-12, 4D * double.Epsilon);
            Assert.True(double.IsFinite(actual) && actual != 0 && Math.Abs(expected - actual) <= tolerance,
                $"{context}: expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}");
        }
    }
}
