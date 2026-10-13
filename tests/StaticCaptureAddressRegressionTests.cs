using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class StaticCaptureAddressRegressionTests(GpuFixture fixture)
{
    private static readonly Value Constant = new(7, '\u8000', true);
    private static Value current;
    private static Value? optional;
    private static GpuView<int> nativeView;

    [Fact]
    public void StaticReadonlyValueAddressesSupportReadonlyInstanceCalls()
    {
        int[] output = new int[2];
        fixture.Gpu.For(0, output.Length, i => output[i] = Constant.Read(i));
        Assert.Equal(new[] { 32775, 32776 }, output);
    }

    [Fact]
    public void StaticScalarAddressCapturesRefreshOnEveryCachedInvocation()
    {
        current = new Value(11, '\uffff', true);
        int[] output = new int[2];
        Action<int> body = i => output[i] = current.Read(i);
        fixture.Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 65546, 65547 }, output);
        current = new Value(17, '?', false);
        fixture.Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 17, 18 }, output);
        Assert.Equal(17, current.Number);
    }

    [Fact]
    public void StaticNullableAddressCapturesDeferAbsentPayloadErrorsAndRecover()
    {
        optional = null;
        int[] output = [41, 43];
        Action<int> body = i => output[i] = optional!.Value.Read(i);
        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, output.Length, body));
        Assert.Equal(new[] { 41, 43 }, output);
        optional = new Value(19, '\ud800', true);
        fixture.Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 55315, 55316 }, output);
    }

    [Fact]
    public void StaticAddressCapturesRejectScalarMutationAndEscapingReferences()
    {
        current = new Value(23, '?', false);
        int[] output = [47];
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            current.Increment();
        }));
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            Increment(ref current);
        }));
        Assert.Equal(23, current.Number);
        Assert.Equal(new[] { 47 }, output);
        fixture.Gpu.For(0, 1, i => output[i] = current.Read(i));
        Assert.Equal(new[] { 23 }, output);
    }

    [Fact]
    public void StaticNativeViewAddressCapturesValidateOwnerAndLifetimeBeforeWrites()
    {
        using var buffer = fixture.Gpu.Upload(new[] { 29, 31 });
        nativeView = buffer.DeviceView;
        int[] output = [53, 59];
        Action<int> body = i =>
        {
            nativeView[i] += 1;
            output[i] = nativeView.Length;
        };
        fixture.Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 30, 32 }, buffer.ToArray());
        Assert.Equal(new[] { 2, 2 }, output);

        using var other = new Gpu(new() { Backend = GpuBackend.Cpu });
        using var foreign = other.Upload(new[] { 61, 67 });
        nativeView = foreign.DeviceView;
        Assert.Throws<ArgumentException>(() => fixture.Gpu.For(0, output.Length, body));
        Assert.Equal(new[] { 61, 67 }, foreign.ToArray());
        Assert.Equal(new[] { 2, 2 }, output);

        nativeView = buffer.DeviceView;
        buffer.Dispose();
        Assert.Throws<ObjectDisposedException>(() => fixture.Gpu.For(0, output.Length, body));
        Assert.Equal(new[] { 2, 2 }, output);
        nativeView = default;
    }

    private static void Increment(ref Value value) => value.Number++;

    private struct Value(int number, char marker, bool enabled)
    {
        public int Number = number;
        public char Marker = marker;
        public bool Enabled = enabled;
        public readonly int Read(int offset) => Number + (Enabled ? Marker : 0) + offset;
        public void Increment() => Number++;
    }
}
