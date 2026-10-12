using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AdditionalTranscendentalTests(GpuFixture fixture)
{
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
    [InlineData(10)]
    [InlineData(11)]
    [InlineData(12)]
    public void UnaryFunctionsMatchClrDomainsSignsAndFiniteValues(int operation)
    {
        double[] input64 = Inputs(operation);
        float[] input32 = input64.Select(value => (float)value).ToArray();
        double[] expected64 = input64.Select(value => CalculateDouble(operation, value)).ToArray();
        float[] expected32 = input32.Select(value => CalculateSingle(operation, value)).ToArray();
        double[] output64 = new double[input64.Length];
        float[] output32 = new float[input32.Length];
        fixture.Gpu.For(0, input64.Length, index =>
        {
            output64[index] = CalculateDouble(operation, input64[index]);
            output32[index] = CalculateSingle(operation, input32[index]);
        });
        for (int i = 0; i < input64.Length; i++)
        {
            AssertDouble(expected64[i], output64[i], $"operation{operation}({input64[i]:R})");
            AssertSingle(expected32[i], output32[i], $"operation{operation}({input32[i]:R})");
        }
    }

    [Fact]
    public void TangentRemainsAccurateAroundSmallArgumentsZerosAndPoles()
    {
        double[] centers = [1e-8, 1e-7, 1e-6, Math.PI / 2, Math.PI, 2 * Math.PI];
        double[] input64 = centers.SelectMany(value => new[]
        {
            Math.BitDecrement(value), value, Math.BitIncrement(value), value - 1e-8, value + 1e-8
        }).SelectMany(value => new[] { value, -value }).ToArray();
        float[] input32 = centers.SelectMany(value => new[]
        {
            MathF.BitDecrement((float)value), (float)value, MathF.BitIncrement((float)value)
        }).SelectMany(value => new[] { value, -value }).ToArray();
        double[] output64 = new double[input64.Length];
        float[] output32 = new float[input32.Length];
        fixture.Gpu.For(0, input64.Length, i => output64[i] = Math.Tan(input64[i]));
        fixture.Gpu.For(0, input32.Length, i => output32[i] = MathF.Tan(input32[i]));
        for (int i = 0; i < input64.Length; i++)
            AssertDouble(Math.Tan(input64[i]), output64[i], $"Tan({input64[i]:R})");
        for (int i = 0; i < input32.Length; i++)
            AssertSingle(MathF.Tan(input32[i]), output32[i], $"Tan({input32[i]:R})");
    }

    [Fact]
    public void Atan2PreservesEverySignedZeroAndInfiniteQuadrant()
    {
        double[] components = [-0.0, 0.0, -1, 1, double.NegativeInfinity, double.PositiveInfinity, double.NaN];
        double[] y = components.SelectMany(value => Enumerable.Repeat(value, components.Length)).ToArray();
        double[] x = Enumerable.Range(0, components.Length).SelectMany(_ => components).ToArray();
        double[] expected64 = y.Zip(x, Math.Atan2).ToArray();
        float[] y32 = y.Select(value => (float)value).ToArray();
        float[] x32 = x.Select(value => (float)value).ToArray();
        float[] expected32 = y32.Zip(x32, MathF.Atan2).ToArray();
        double[] output64 = new double[y.Length];
        float[] output32 = new float[y.Length];
        fixture.Gpu.For(0, y.Length, index =>
        {
            output64[index] = Math.Atan2(y[index], x[index]);
            output32[index] = MathF.Atan2(y32[index], x32[index]);
        });
        for (int i = 0; i < y.Length; i++)
        {
            AssertDouble(expected64[i], output64[i], $"Atan2({y[i]:R},{x[i]:R})");
            AssertSingle(expected32[i], output32[i], $"Atan2({y32[i]:R},{x32[i]:R})");
        }
    }

    [Fact]
    public void SinCosTupleSupportsDeconstructionAndKeepsSignedZeroAndHugeAngles()
    {
        double[] input64 = [-0.0, 0, double.Epsilon, -0.5, 0.5, Math.PI / 2, 1e30,
            double.MaxValue, double.NegativeInfinity, double.PositiveInfinity, double.NaN];
        float[] input32 = input64.Select(value => (float)value).ToArray();
        double[] sin64 = new double[input64.Length], cos64 = new double[input64.Length];
        float[] sin32 = new float[input32.Length], cos32 = new float[input32.Length];
        fixture.Gpu.For(0, input64.Length, index =>
        {
            (sin64[index], cos64[index]) = Math.SinCos(input64[index]);
            var pair = MathF.SinCos(input32[index]);
            sin32[index] = pair.Sin;
            cos32[index] = pair.Cos;
        });
        for (int i = 0; i < input64.Length; i++)
        {
            AssertDouble(Math.Sin(input64[i]), sin64[i], "SinCos.Sin", absoluteTolerance: 2e-14);
            AssertDouble(Math.Cos(input64[i]), cos64[i], "SinCos.Cos", absoluteTolerance: 2e-14);
            AssertSingle(MathF.Sin(input32[i]), sin32[i], "SinCos.Sin", absoluteTolerance: 2e-7);
            AssertSingle(MathF.Cos(input32[i]), cos32[i], "SinCos.Cos", absoluteTolerance: 2e-7);
        }
    }

    private static double[] Inputs(int operation) => operation switch
    {
        0 => [-double.MaxValue, -1e300, -1e30, -Math.PI, -Math.PI / 2, -0.5, -double.Epsilon,
            -0.0, 0.0, double.Epsilon, (double)float.Epsilon, 0.5, Math.PI / 2, Math.PI, 1e30, 1e300,
            double.MaxValue, double.NegativeInfinity, double.PositiveInfinity, double.NaN],
        1 or 2 or 9 => [-2, -1, Math.BitIncrement(-1), -0.5, -1e-8, -double.Epsilon, -0.0, 0.0,
            double.Epsilon, (double)float.Epsilon, 1e-8, 0.5, Math.BitDecrement(1), 1, 2,
            double.NegativeInfinity, double.PositiveInfinity, double.NaN],
        8 => [-1, -0.0, 0, 0.5, 1, Math.BitIncrement(1), 1.0001, 2, 10, 1e154, 1e300,
            double.MaxValue, double.NegativeInfinity, double.PositiveInfinity, double.NaN],
        11 or 12 => [-1, -0.0, 0.0, double.Epsilon, BitConverter.Int64BitsToDouble(0x000fffffffffffff),
            Math.ScaleB(1, -1050), (double)float.Epsilon, 0.125, 0.5, 1, 2, 10, 100, 1e100,
            double.MaxValue, double.NegativeInfinity, double.PositiveInfinity, double.NaN],
        _ => [-double.MaxValue, -1e300, -710.6, -710, -100, -20, -1, -1e-8, -double.Epsilon,
            -0.0, 0, double.Epsilon, (double)float.Epsilon, 1e-8, 1, 20, 100, 710, 710.6, 1e154,
            1e300, double.MaxValue, double.NegativeInfinity, double.PositiveInfinity, double.NaN]
    };

    private static double CalculateDouble(int operation, double value) => operation switch
    {
        0 => Math.Tan(value), 1 => Math.Asin(value), 2 => Math.Acos(value), 3 => Math.Atan(value),
        4 => Math.Sinh(value), 5 => Math.Cosh(value), 6 => Math.Tanh(value), 7 => Math.Asinh(value),
        8 => Math.Acosh(value), 9 => Math.Atanh(value), 10 => Math.Cbrt(value),
        11 => Math.Log2(value), _ => Math.Log10(value)
    };

    private static float CalculateSingle(int operation, float value) => operation switch
    {
        0 => MathF.Tan(value), 1 => MathF.Asin(value), 2 => MathF.Acos(value), 3 => MathF.Atan(value),
        4 => MathF.Sinh(value), 5 => MathF.Cosh(value), 6 => MathF.Tanh(value), 7 => MathF.Asinh(value),
        8 => MathF.Acosh(value), 9 => MathF.Atanh(value), 10 => MathF.Cbrt(value),
        11 => MathF.Log2(value), _ => MathF.Log10(value)
    };

    private static void AssertDouble(double expected, double actual, string context, double absoluteTolerance = 0)
    {
        if (double.IsNaN(expected))
            Assert.True(double.IsNaN(actual), $"{context}: expected NaN, actual {actual:R}");
        else if (double.IsInfinity(expected) || expected == 0)
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual),
                $"{context}: expected {expected:R}, actual {actual:R}");
        else
        {
            double tolerance = Math.Max(absoluteTolerance, Math.Max(Math.Abs(expected) * 4e-12, 4 * double.Epsilon));
            Assert.True(double.IsFinite(actual) && actual != 0 && Math.Abs(expected - actual) <= tolerance,
                $"{context}: expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}");
        }
    }

    private static void AssertSingle(float expected, float actual, string context, double absoluteTolerance = 0)
    {
        if (float.IsNaN(expected))
            Assert.True(float.IsNaN(actual), $"{context}: expected NaN, actual {actual:R}");
        else if (float.IsInfinity(expected) || expected == 0)
            Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
                $"{context}: expected {expected:R}, actual {actual:R}");
        else
        {
            double tolerance = Math.Max(absoluteTolerance, Math.Max(Math.Abs((double)expected) * 4e-6, 4 * (double)float.Epsilon));
            Assert.True(float.IsFinite(actual) && actual != 0 && Math.Abs((double)expected - actual) <= tolerance,
                $"{context}: expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}");
        }
    }
}
