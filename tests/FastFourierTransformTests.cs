using System.Numerics;
using FastFourierTransform;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class FastFourierTransformTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(4, 8, true)]
    [InlineData(8, 2, false)]
    [InlineData(1, 8, true)]
    [InlineData(4, 1, false)]
    [InlineData(1, 1, true)]
    public void ForwardMatchesDirectDftAndPreservesInput(int rows, int columns, bool normalized)
    {
        var random = new Random(31);
        var input = new Complex[rows, columns];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                input[row, column] = new Complex(2 * random.NextDouble() - 1, 2 * random.NextDouble() - 1);
        var original = (Complex[,])input.Clone();

        Complex[,] expected = DirectForwardDft(input, normalized);
        Complex[,] actual = FourierTransform.Forward2D(fixture.Gpu, input, normalized);

        Assert.NotSame(input, actual);
        Assert.Equal(rows, actual.GetLength(0));
        Assert.Equal(columns, actual.GetLength(1));
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                AssertClose(expected[row, column], actual[row, column]);
                Assert.Equal(original[row, column], input[row, column]);
            }
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OriginImpulseHasAFlatSpectrum(bool normalized)
    {
        var input = new Complex[2, 8];
        input[0, 0] = new Complex(2, -3);

        Complex[,] actual = FourierTransform.Forward2D(fixture.Gpu, input, normalized);
        Complex expected = input[0, 0] * (normalized ? 1 / Math.Sqrt(input.Length) : 1);

        foreach (Complex value in actual)
            AssertClose(expected, value);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ConstantInputHasOnlyTheDcComponent(bool normalized)
    {
        var input = new Complex[4, 8];
        var value = new Complex(-2, 0.75);
        for (int row = 0; row < input.GetLength(0); row++)
            for (int column = 0; column < input.GetLength(1); column++)
                input[row, column] = value;

        Complex[,] actual = FourierTransform.Forward2D(fixture.Gpu, input, normalized);
        Complex dc = value * (normalized ? Math.Sqrt(input.Length) : input.Length);

        for (int row = 0; row < actual.GetLength(0); row++)
            for (int column = 0; column < actual.GetLength(1); column++)
                AssertClose(row == 0 && column == 0 ? dc : Complex.Zero, actual[row, column]);
    }

    [Fact]
    public void PositiveComplexFrequencyMapsToTheSameForwardFrequency()
    {
        const int rows = 4;
        const int columns = 8;
        var input = new Complex[rows, columns];
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                double angle = 2 * Math.PI * ((double)row / rows + 3.0 * column / columns);
                input[row, column] = new Complex(Math.Cos(angle), Math.Sin(angle));
            }
        }

        Complex[,] actual = FourierTransform.Forward2D(fixture.Gpu, input, normalized: false);

        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                AssertClose(row == 1 && column == 3 ? new Complex(rows * columns, 0) : Complex.Zero, actual[row, column]);
    }

    [Fact]
    public void NormalizationIsEnabledByDefault()
    {
        var input = new Complex[2, 2];
        input[0, 0] = Complex.One;

        Complex[,] actual = FourierTransform.Forward2D(fixture.Gpu, input);

        foreach (Complex value in actual)
            AssertClose(new Complex(0.5, 0), value);
    }

    [Fact]
    public void RejectsNullArgumentsAndEmptyAxes()
    {
        Assert.Throws<ArgumentNullException>(() => FourierTransform.Forward2D(null!, new Complex[1, 1]));
        Assert.Throws<ArgumentNullException>(() => FourierTransform.Forward2D(fixture.Gpu, null!));
        Assert.Throws<ArgumentException>(() => FourierTransform.Forward2D(fixture.Gpu, new Complex[0, 3]));
        Assert.Throws<ArgumentException>(() => FourierTransform.Forward2D(fixture.Gpu, new Complex[3, 0]));
    }

    [Theory]
    [InlineData(3, 4)]
    [InlineData(4, 3)]
    [InlineData(6, 8)]
    public void RejectsAxesThatAreNotPowersOfTwo(int rows, int columns)
    {
        var error = Assert.Throws<ArgumentException>(() =>
            FourierTransform.Forward2D(fixture.Gpu, new Complex[rows, columns]));

        Assert.Equal("input", error.ParamName);
    }

    private static Complex[,] DirectForwardDft(Complex[,] input, bool normalized)
    {
        int rows = input.GetLength(0);
        int columns = input.GetLength(1);
        var output = new Complex[rows, columns];
        double scale = normalized ? 1 / Math.Sqrt(input.Length) : 1;
        for (int frequencyRow = 0; frequencyRow < rows; frequencyRow++)
        {
            for (int frequencyColumn = 0; frequencyColumn < columns; frequencyColumn++)
            {
                Complex sum = Complex.Zero;
                for (int row = 0; row < rows; row++)
                {
                    for (int column = 0; column < columns; column++)
                    {
                        double angle = -2 * Math.PI * ((double)frequencyRow * row / rows
                            + (double)frequencyColumn * column / columns);
                        sum += input[row, column] * new Complex(Math.Cos(angle), Math.Sin(angle));
                    }
                }
                output[frequencyRow, frequencyColumn] = sum * scale;
            }
        }
        return output;
    }

    private static void AssertClose(Complex expected, Complex actual)
    {
        double tolerance = 1e-8 * (1 + expected.Magnitude);
        Assert.InRange(Complex.Abs(actual - expected), 0, tolerance);
    }
}
