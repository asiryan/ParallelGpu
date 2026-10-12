using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class GpuViewCaptureTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void BufferViewsSupportHelpersPassThroughReferencesAndAtomicOperations()
    {
        using var values = Gpu.Upload(new[] { 2, 4, 6, 8 });
        using var counter = Gpu.Allocate<int>(1);
        GpuView<int> view = values.DeviceView;
        GpuView<int> counterView = counter.DeviceView;
        int[] observed = new int[values.Length];

        Gpu.For(0, observed.Length, i =>
        {
            GpuView<int> current = PassThrough(view);
            ref int element = ref Element(current, i, 1);
            Add(ref element, current.Length);
            GpuAtomic.Add(ref counterView[0], 1);
            observed[i] = Read(current, i);
        });

        Assert.Equal(new[] { 6, 8, 10, 12 }, values.ToArray());
        Assert.Equal(new[] { 6, 8, 10, 12 }, observed);
        Assert.Equal(4, counter.ToArray()[0]);
    }

    [Fact]
    public void NumericPayloadsUseTheirNativeRepresentation()
    {
        using var values = Gpu.Upload(new[]
        {
            new GpuViewNumericPayload { Number = 7, Weight = 1.5f },
            new GpuViewNumericPayload { Number = 9, Weight = 2.5f }
        });
        GpuView<GpuViewNumericPayload> view = values.DeviceView;
        float[] observed = new float[2];

        Gpu.For(0, observed.Length, i =>
        {
            ref GpuViewNumericPayload element = ref view[i];
            element.Number += 3;
            element.Weight *= 2;
            observed[i] = element.Number + element.Weight;
        });

        Assert.Equal(new[] { 13f, 17f }, observed);
        Assert.Equal(new[] { 10, 12 }, values.ToArray().Select(value => value.Number));
        Assert.Equal(new[] { 3f, 5f }, values.ToArray().Select(value => value.Weight));
    }

    [Fact]
    public void CachedBodiesRefreshViewCapturesAndValidateEveryInvocation()
    {
        using var first = Gpu.Upload(new[] { 3, 4 });
        using var second = Gpu.Upload(new[] { 7, 8 });
        GpuView<int> view = first.DeviceView;
        int[] observed = [-1, -1];
        Action<int> body = i => { view[i] += 2; observed[i] = view[i]; };

        Gpu.For(0, observed.Length, body);
        Assert.Equal(new[] { 5, 6 }, observed);
        view = second.DeviceView;
        Gpu.For(0, observed.Length, body);
        Assert.Equal(new[] { 9, 10 }, observed);
        Assert.Equal(new[] { 5, 6 }, first.ToArray());

        view = default;
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, observed.Length, body));
        Assert.Equal(new[] { 9, 10 }, observed);

        view = second.DeviceView;
        Gpu.For(0, observed.Length, body);
        Assert.Equal(new[] { 11, 12 }, observed);
    }

    [Fact]
    public void DefaultDisposedAndForeignViewsFailBeforeHostOrDeviceWrites()
    {
        using var output = Gpu.Upload(new[] { 41 });
        GpuView<int> outputView = output.DeviceView;
        int[] host = [43];
        GpuView<int> invalid = default;
        Action<int> body = i => { outputView[i] = 99; host[i] = invalid.Length; };

        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, 1, body));
        Assert.Equal(41, output.ToArray()[0]);
        Assert.Equal(43, host[0]);

        var expired = Gpu.Upload(new[] { 7 });
        invalid = expired.DeviceView;
        expired.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, 1, body));
        Assert.Equal(41, output.ToArray()[0]);
        Assert.Equal(43, host[0]);

        using var other = new Gpu(new GpuOptions { Backend = GpuBackend.Cpu });
        using var foreign = other.Upload(new[] { 8 });
        invalid = foreign.DeviceView;
        Assert.Throws<ArgumentException>(() => Gpu.For(0, 1, body));
        Assert.Equal(41, output.ToArray()[0]);
        Assert.Equal(43, host[0]);
    }

    [Fact]
    public void EmptyBufferViewsRetainTheirValidOwnerAndZeroLength()
    {
        using var values = Gpu.Allocate<int>(0);
        GpuView<int> view = values.DeviceView;
        int[] observed = [-1];
        Gpu.For(0, 1, i => observed[i] = view.Length);
        Assert.Equal(0, observed[0]);
    }

    [Fact]
    public void CopiedViewsShareStorageAndExpireTogetherWithTheBuffer()
    {
        using var values = Gpu.Upload(new[] { 2, 4 });
        GpuView<int> first = values.DeviceView;
        GpuView<int> copied = first;
        GpuView<int> reacquired = values.DeviceView;
        int[] observed = [-1, -1];
        Gpu.For(0, observed.Length, i =>
        {
            first[i] += 1;
            copied[i] += 2;
            observed[i] = reacquired[i];
        });
        Assert.Equal(new[] { 5, 7 }, observed);
        Assert.Equal(observed, values.ToArray());

        values.Dispose();
        Assert.Throws<ObjectDisposedException>(() => values.DeviceView);
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, observed.Length, i => observed[i] = copied[i]));
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, observed.Length, i => observed[i] = reacquired[i]));
        Assert.Equal(new[] { 5, 7 }, observed);
    }

    [Fact]
    public void DisposingTheExecutorRevokesPreviouslyBorrowedViews()
    {
        using var owner = new Gpu(new GpuOptions { Backend = GpuBackend.Cpu });
        using var values = owner.Allocate<int>(1);
        GpuView<int> view = values.DeviceView;
        owner.Dispose();
        Assert.Throws<ObjectDisposedException>(() => values.DeviceView);
        int[] observed = [41];
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, 1, i => observed[i] = view.Length));
        Assert.Equal(41, observed[0]);
    }

    [Fact]
    public void CapturedViewDescriptorsCannotBeReassignedThroughReferences()
    {
        using var values = Gpu.Upload(new[] { 3 });
        GpuView<int> view = values.DeviceView;
        int[] observed = [41];

        Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i =>
        {
            observed[i] = 99;
            ref GpuView<int> descriptor = ref view;
            descriptor = default;
        }));

        Assert.Equal(41, observed[0]);
        Assert.Equal(3, values.ToArray()[0]);
    }

    [Fact]
    public void PayloadsNeedingAutomaticConversionAreRejectedBeforeWrites()
    {
        GpuView<GpuViewBooleanPayload> view = default;
        int[] observed = [41];

        var failure = Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i =>
        {
            observed[i] = 99;
            observed[i] = view[i].Enabled ? 1 : 0;
        }));

        Assert.Contains("automatic value conversion", failure.Message);
        Assert.Equal(41, observed[0]);
    }

    private static GpuView<int> PassThrough(GpuView<int> view) => view;
    private static ref int Element(GpuView<int> view, int index, int divisor) => ref view[index / divisor];
    private static int Read(GpuView<int> view, int index) => view[index];
    private static void Add(ref int value, int increment) => value += increment;
}

public struct GpuViewNumericPayload
{
    public int Number;
    public float Weight;
}

public struct GpuViewBooleanPayload
{
    public bool Enabled;
}
