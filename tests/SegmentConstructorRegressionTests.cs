using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class SegmentConstructorRegressionTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(-1, 0)]
    [InlineData(0, -1)]
    [InlineData(-1, -1)]
    [InlineData(4, -1)]
    [InlineData(4, 0)]
    [InlineData(2, 2)]
    [InlineData(0, 4)]
    [InlineData(3, 1)]
    [InlineData(int.MaxValue, 0)]
    [InlineData(0, int.MaxValue)]
    [InlineData(int.MaxValue, int.MaxValue)]
    public void InvalidConstructorArgumentsMatchClrAndRollBackWrites(int offset, int count)
    {
        int[] data = [7, 11, 17];
        int[] output = [41];
        var hostFailure = Record.Exception(() => _ = new ArraySegment<int>(data, offset, count));
        var deviceFailure = Record.Exception(() => fixture.Gpu.For(0, 1, _ =>
        {
            data[0] = 99;
            output[0] = 99;
            var segment = new ArraySegment<int>(data, offset, count);
            output[0] = segment.Count;
        }));

        AssertMatchingFailure(hostFailure, deviceFailure);
        Assert.Equal(new[] { 7, 11, 17 }, data);
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(-1, -1)]
    [InlineData(int.MaxValue, int.MaxValue)]
    public void NullArrayTakesPrecedenceOverInvalidOffsetAndCount(int offset, int count)
    {
        int[]? data = null;
        int[] output = [41];
        var hostFailure = Record.Exception(() => _ = new ArraySegment<int>(data!, offset, count));
        var deviceFailure = Record.Exception(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            var segment = new ArraySegment<int>(data!, offset, count);
            output[0] = segment.Count;
        }));

        AssertMatchingFailure(hostFailure, deviceFailure);
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Fact]
    public void ConstructorInHelperPreservesRangeExceptionAndRecovery()
    {
        int[] data = [7, 11, 17];
        int[] output = [41];
        var hostFailure = Record.Exception(() => _ = MakeSegment(data, 2, 2));
        var deviceFailure = Record.Exception(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = MakeSegment(data, 2, 2).Count;
        }));

        AssertMatchingFailure(hostFailure, deviceFailure);
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 3)]
    [InlineData(1, 2)]
    [InlineData(3, 0)]
    public void ValidConstructorBoundariesAndHelperViewsMatchClr(int offset, int count)
    {
        int[] data = [7, 11, 17];
        var expected = new ArraySegment<int>(data, offset, count);
        int[] output = new int[3];
        fixture.Gpu.For(0, 1, _ =>
        {
            var segment = new ArraySegment<int>(data, offset, count);
            var throughHelper = MakeSegment(data, offset, count);
            output[0] = segment.Count;
            output[1] = segment.Count == 0 ? -1 : segment[0];
            output[2] = throughHelper.Count == 0 ? -1 : throughHelper[count - 1];
        });

        Assert.Equal(new[] { expected.Count, count == 0 ? -1 : expected[0],
            count == 0 ? -1 : expected[count - 1] }, output);
    }

    [Fact]
    public void EmptyArrayAcceptsZeroOffsetAndCount()
    {
        int[] data = [];
        int[] output = [41];
        fixture.Gpu.For(0, 1, _ => output[0] = new ArraySegment<int>(data, 0, 0).Count);
        Assert.Equal(new ArraySegment<int>(data, 0, 0).Count, output[0]);
    }

    private static ArraySegment<int> MakeSegment(int[] data, int offset, int count) =>
        new(data, offset, count);

    private static void AssertMatchingFailure(Exception? expected, Exception? actual)
    {
        Assert.NotNull(expected);
        Assert.NotNull(actual);
        Assert.Equal(expected.GetType(), actual.GetType());
        Assert.Equal(Assert.IsAssignableFrom<ArgumentException>(expected).ParamName,
            Assert.IsAssignableFrom<ArgumentException>(actual).ParamName);
    }

    private void Recover(int[] output)
    {
        fixture.Gpu.For(0, output.Length, i => output[i] = 17 + i);
        Assert.Equal(Enumerable.Range(0, output.Length).Select(i => 17 + i), output);
    }
}
