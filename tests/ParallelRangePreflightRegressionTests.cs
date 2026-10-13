using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ParallelRangePreflightRegressionTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void ConfiguredExcessiveRangesRejectBeforeDeviceSelection(bool typed, bool invalidIndex)
    {
        var options = invalidIndex
            ? new GpuOptions { Backend = GpuBackend.Cpu, DeviceIndex = int.MaxValue }
            : new GpuOptions { Backend = (GpuBackend)int.MaxValue };
        int[] output = [41];
        Action<int> body = i => output[0] = i;

        var error = typed
            ? Assert.Throws<ArgumentOutOfRangeException>(() =>
                GpuParallel.For<int>(int.MinValue, int.MaxValue, options, body))
            : Assert.Throws<ArgumentOutOfRangeException>(() =>
                GpuParallel.For(int.MinValue, int.MaxValue, options, body));
        Assert.Equal("toExclusive", error.ParamName);
        Assert.Equal(new[] { 41 }, output);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DefaultExcessiveRangesRejectBeforeInitializingAnExecutor(bool typed)
    {
        int[] output = [41];
        Action<int> body = i => output[0] = i;
        var error = typed
            ? Assert.Throws<ArgumentOutOfRangeException>(() =>
                GpuParallel.For<int>(int.MinValue, int.MaxValue, body))
            : Assert.Throws<ArgumentOutOfRangeException>(() =>
                GpuParallel.For(int.MinValue, int.MaxValue, body));
        Assert.Equal("toExclusive", error.ParamName);
        Assert.Equal(new[] { 41 }, output);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmptyAndReversedRangesSkipInvalidDeviceOptions(bool typed)
    {
        var options = new GpuOptions { Backend = (GpuBackend)int.MaxValue, DeviceIndex = -1 };
        int[] output = [41];
        Action<int> body = i => output[0] = i;
        if (typed)
        {
            GpuParallel.For<int>(7, 7, options, body);
            GpuParallel.For<int>(7, -7, options, body);
            GpuParallel.For<int>(7, 7, body);
            GpuParallel.For<int>(7, -7, body);
        }
        else
        {
            GpuParallel.For(7, 7, options, body);
            GpuParallel.For(7, -7, options, body);
            GpuParallel.For(7, 7, body);
            GpuParallel.For(7, -7, body);
        }
        Assert.Equal(new[] { 41 }, output);
    }

    [Fact]
    public void NullArgumentsKeepTheirPrecedenceOverExcessiveRanges()
    {
        var options = new GpuOptions { Backend = (GpuBackend)int.MaxValue };
        int[] output = [41];
        Action<int> body = i => output[0] = i;
        Assert.Equal("body", Assert.Throws<ArgumentNullException>(() =>
            GpuParallel.For(int.MinValue, int.MaxValue, null!)).ParamName);
        Assert.Equal("body", Assert.Throws<ArgumentNullException>(() =>
            GpuParallel.For<string>(int.MinValue, int.MaxValue, null!)).ParamName);
        Assert.Equal("options", Assert.Throws<ArgumentNullException>(() =>
            GpuParallel.For(int.MinValue, int.MaxValue, null!, body)).ParamName);
        Assert.Equal("options", Assert.Throws<ArgumentNullException>(() =>
            GpuParallel.For<string>(int.MinValue, int.MaxValue, null!, body)).ParamName);
        Assert.Equal("body", Assert.Throws<ArgumentNullException>(() =>
            GpuParallel.For(int.MinValue, int.MaxValue, options, null!)).ParamName);
        Assert.Equal("body", Assert.Throws<ArgumentNullException>(() =>
            GpuParallel.For<string>(int.MinValue, int.MaxValue, options, null!)).ParamName);
    }

    [Fact]
    public void TypedElementValidationStillPrecedesRangeAndDeviceValidation()
    {
        var options = new GpuOptions { Backend = (GpuBackend)int.MaxValue };
        int[] output = [41];
        Action<int> body = i => output[0] = i;
        Assert.Throws<NotSupportedException>(() =>
            GpuParallel.For<string>(int.MinValue, int.MaxValue, body));
        Assert.Throws<NotSupportedException>(() =>
            GpuParallel.For<string>(int.MinValue, int.MaxValue, options, body));
        Assert.Throws<NotSupportedException>(() => GpuParallel.For<string>(7, 7, body));
        Assert.Throws<NotSupportedException>(() => GpuParallel.For<string>(7, -7, options, body));
        Assert.Equal(new[] { 41 }, output);
    }
}
