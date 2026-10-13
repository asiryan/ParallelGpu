using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class OptimizationSafetyCaptureTests(GpuFixture fixture)
{
    [Fact]
    public void AdapterCallbacksCannotInvalidateBorrowedNativeViewsBeforeTheLaunch()
    {
        DisposingCapture.Register();
        using var original = fixture.Gpu.Upload(new[] { 17 });
        var native = original.DeviceView;
        var adapted = new DisposingCapture([7]);
        int[] output = [41], journal = [43];
        Action<int> body = i =>
        {
            journal[i] = 99;
            adapted[i]++;
            // Reading Length safely demonstrates stale lifetime validation without
            // dereferencing a pointer freed by the host adapter callback.
            output[i] = native.Length + adapted[i];
        };

        fixture.Gpu.For(0, 1, body);
        Assert.Equal(9, output[0]);
        Assert.Equal(99, journal[0]);
        Assert.Equal(8, adapted.Values[0]);
        Assert.Equal(1, adapted.CommitCount);

        output[0] = 41;
        journal[0] = 43;
        adapted.BeforeCapture = original.Dispose;
        Assert.Throws<ObjectDisposedException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        Assert.Equal(43, journal[0]);
        Assert.Equal(8, adapted.Values[0]);
        Assert.Equal(1, adapted.CommitCount);

        using var replacement = fixture.Gpu.Upload(new[] { 19, 23 });
        native = replacement.DeviceView;
        adapted.BeforeCapture = null;
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(11, output[0]);
        Assert.Equal(99, journal[0]);
        Assert.Equal(9, adapted.Values[0]);
        Assert.Equal(2, adapted.CommitCount);
    }

    [Fact]
    public void DisposalInALaterAdapterCallbackIsDetectedAfterEarlierNativeValidation()
    {
        DisposingCapture.Register();
        using var borrowed = fixture.Gpu.Upload(new[] { 17 });
        var native = borrowed.DeviceView;
        var first = new DisposingCapture([7]);
        var second = new DisposingCapture([11]) { BeforeCapture = borrowed.Dispose };
        int[] output = [41];
        Assert.Throws<ObjectDisposedException>(() => fixture.Gpu.For(0, 1, i =>
        {
            output[i] = 99;
            first[i]++;
            second[i]++;
            output[i] = native.Length + first[i] + second[i];
        }));
        Assert.Equal(41, output[0]);
        Assert.Equal(7, first.Values[0]);
        Assert.Equal(11, second.Values[0]);
        Assert.Equal(0, first.CommitCount);
        Assert.Equal(0, second.CommitCount);
    }

    private sealed class DisposingCapture(int[] values)
    {
        private static readonly Lazy<bool> registration = new(() =>
        {
            GpuCollectionAdapters.Register<DisposingCapture, int>(source =>
            {
                source.BeforeCapture?.Invoke();
                return new(source.Values, commit: _ => source.CommitCount++, identity: source);
            });
            return true;
        });

        public static void Register() => _ = registration.Value;
        public int[] Values { get; } = values;
        public Action? BeforeCapture { get; set; }
        public int CommitCount { get; private set; }
        public int this[int index]
        {
            get => Values[index];
            set => Values[index] = value;
        }
    }
}
