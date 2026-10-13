using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class HelperPreflightRegressionTests(GpuFixture fixture)
{
    private static readonly NativeClassCarrier NativeHolder = new();
    private static readonly NumberHolder ManagedHolder = new(17);

    [Fact]
    public void ReadonlyStaticClassFieldsCannotHideNativeViewsInsideScalarHelpers()
    {
        int[] output = [41];
        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, index =>
        {
            output[index] = 99;
            output[index] = ReadHiddenClassViewLength();
        }));

        Assert.Contains("static", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Fact]
    public void ReadonlyStaticClassFieldsCannotReturnViewsThroughTransitiveHelpers()
    {
        int[] output = [43];
        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, index =>
        {
            output[index] = 99;
            output[index] = ReturnHiddenClassView().Length;
        }));

        Assert.Contains("static", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(43, output[0]);
        Recover(output);
    }

    [Fact]
    public void ReadonlyStaticManagedObjectsDoNotBecomeUnvalidatedHelperConstants()
    {
        int[] output = [47];
        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, index =>
        {
            output[index] = 99;
            output[index] = ReadHiddenNumber();
        }));

        Assert.Contains("static", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(47, output[0]);
        Assert.Equal(17, ManagedHolder.Number);
        Recover(output);
    }

    [Fact]
    public void ManagedConstructorsInsideScalarHelpersAreRejectedBeforeBackendCompilation()
    {
        int[] output = [53];
        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, index =>
        {
            output[index] = 99;
            output[index] = AllocateManagedNumber(index);
        }));

        Assert.Null(failure.InnerException);
        Assert.Equal(53, output[0]);
        Recover(output);
    }

    private void Recover(int[] output)
    {
        fixture.Gpu.For(0, output.Length, index => output[index] = index + 23);
        Assert.Equal(new[] { 23 }, output);
    }

    private static int ReadHiddenClassViewLength() => NativeHolder.View.Length;
    private static GpuView<int> ReturnHiddenClassView() => InnerHiddenClassView();
    private static GpuView<int> InnerHiddenClassView() => NativeHolder.View;
    private static int ReadHiddenNumber() => ManagedHolder.Read();
    private static int AllocateManagedNumber(int value) => new NumberHolder(value).Number;

    private sealed class NativeClassCarrier
    {
        public readonly GpuView<int> View = default;
    }

    private sealed class NumberHolder(int number)
    {
        public readonly int Number = number;

        public int Read() => Number;
    }
}
