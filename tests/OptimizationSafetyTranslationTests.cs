using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class OptimizationSafetyTranslationTests(GpuFixture fixture)
{
    [Fact]
    public void NativeOnlyRefReturningHelpersReserveWideFailureScratchAndRecover()
    {
        WideNativePayload[] initial = [new()
        {
            First = 17, Second = 19, Third = 23, Fourth = 29,
            Fifth = 31, Sixth = 37, Seventh = 41, Last = 43,
        }];
        using var buffer = fixture.Gpu.Upload(initial);
        var view = buffer.DeviceView;
        int[] denominator = [0], output = [41];
        Action<int> body = i =>
        {
            output[i] = 99;
            ref WideNativePayload value = ref NativeElementAfterDivision(view, denominator[i]);
            value.First++;
            output[i] = (int)value.Last;
        };
        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(initial, buffer.ToArray());

        denominator[0] = 2;
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(new[] { 43 }, output);
        var updated = buffer.ToArray();
        Assert.Equal(18, updated[0].First);
        Assert.Equal(43, updated[0].Last);
        Assert.Equal(41, updated[0].Seventh);
    }

    [Fact]
    public void SharedTemplatesKeepFreshMappedCapturesAndOwnerContextsAcrossBackends()
    {
        using var other = new Gpu(new() { Backend = GpuBackend.Cpu });
        using var first = fixture.Gpu.Upload(new[] { 7 });
        using var second = other.Upload(new[] { 11 });
        int[] firstOutput = [41], secondOutput = [43];
        var failing = SharedNativeBody(first.DeviceView, firstOutput,
            new() { Enabled = true, Marker = 'A', Amount = null });
        var succeeding = SharedNativeBody(second.DeviceView, secondOutput,
            new() { Enabled = true, Marker = 'C', Amount = 3 });
        Assert.Equal(failing.Method, succeeding.Method);

        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, 1, failing));
        Assert.Equal(new[] { 41 }, firstOutput);
        Assert.Equal(new[] { 7 }, first.ToArray());
        other.For(0, 1, succeeding);
        Assert.Equal(new[] { 81 }, secondOutput);
        Assert.Equal(new[] { 81 }, second.ToArray());
        other.Dispose();

        var recovered = SharedNativeBody(first.DeviceView, firstOutput,
            new() { Enabled = true, Marker = 'B', Amount = 2 });
        Assert.Equal(failing.Method, recovered.Method);
        fixture.Gpu.For(0, 1, recovered);
        Assert.Equal(new[] { 75 }, firstOutput);
        Assert.Equal(new[] { 75 }, first.ToArray());
    }

    private static ref T NativeElementAfterDivision<T>(GpuView<T> view, int denominator) where T : unmanaged
    {
        int offset = 8 / denominator - 4;
        return ref view[offset];
    }

    private static Action<int> SharedNativeBody(GpuView<int> view, int[] output, MappedCapture state) => i =>
    {
        int addition = state.Enabled ? state.Amount!.Value : 0;
        view[i] += addition + state.Marker;
        output[i] = view[i];
    };

    private struct MappedCapture
    {
        public bool Enabled;
        public char Marker;
        public int? Amount;
    }

    public struct WideNativePayload
    {
        public long First;
        public long Second;
        public long Third;
        public long Fourth;
        public long Fifth;
        public long Sixth;
        public long Seventh;
        public long Last;
    }
}
