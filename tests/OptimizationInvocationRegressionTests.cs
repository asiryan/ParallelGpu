using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class OptimizationInvocationRegressionTests(GpuFixture fixture, ITestOutputHelper output)
{
    [Fact]
    public void CachedNativeInvocationsRefreshIdentityMappedAndConvertedCaptures()
    {
        using var buffer = fixture.Gpu.Allocate<long>(1);
        var view = buffer.DeviceView;
        int number = 3;
        double scale = 4.25;
        char marker = 'A';
        bool enabled = true;
        int? optional = 7;
        CaptureMode mode = CaptureMode.First;
        var pair = new IdentityPair { First = 11, Second = 13 };
        var mapped = new MappedPair { Enabled = true, Marker = 'B' };
        Action<int> body = i => view[i] = number + (long)scale + (enabled ? marker : 0) +
            optional.GetValueOrDefault() + (int)mode + pair.First + pair.Second +
            (mapped.Enabled ? mapped.Marker : 0);
        for (int i = 0; i < 20; i++)
            fixture.Gpu.For(0, 1, body);

        long allocated = GC.GetAllocatedBytesForCurrentThread();
        var timer = Stopwatch.StartNew();
        for (int i = 0; i < 256; i++)
        {
            number = i;
            scale = i + 0.25;
            marker = (char)('A' + i % 26);
            enabled = (i & 1) == 0;
            optional = i % 3 == 0 ? null : i + 7;
            mode = (i & 1) == 0 ? CaptureMode.First : CaptureMode.Second;
            pair = new() { First = i + 11, Second = i + 13 };
            mapped = new() { Enabled = (i & 1) != 0, Marker = (char)('a' + i % 26) };
            fixture.Gpu.For(0, 1, body);
        }
        timer.Stop();
        allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
        output.WriteLine($"{fixture.Gpu.Backend}: warmed 256 invocations; {allocated / 256.0:F0} host bytes/call; {timer.Elapsed.TotalMilliseconds / 256:F4} ms/call (informational, no timing assertion)");
        long expected = number + (long)scale + (enabled ? marker : 0) + optional.GetValueOrDefault() +
            (int)mode + pair.First + pair.Second + (mapped.Enabled ? mapped.Marker : 0);
        Assert.Equal(expected, buffer.ToArray()[0]);

        number = -7;
        scale = -3.75;
        enabled = true;
        optional = 19;
        pair = new() { First = 23, Second = 29 };
        mapped = new() { Enabled = false, Marker = char.MaxValue };
        fixture.Gpu.For(0, 1, body);
        expected = number + (long)scale + marker + optional.Value + (int)mode + pair.First + pair.Second;
        Assert.Equal(expected, buffer.ToArray()[0]);
    }

    public enum CaptureMode { First = 17, Second = 31 }

    public struct IdentityPair
    {
        public int First;
        public long Second;
    }

    private struct MappedPair
    {
        public bool Enabled;
        public char Marker;
    }
}
