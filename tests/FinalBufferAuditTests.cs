using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class FinalBufferAuditTests(GpuFixture fixture)
{
    [Fact]
    public void DisposedDownloadRejectsBeforeAllocatingTheFullHostDestination()
    {
        var buffer = fixture.Gpu.Allocate<int>(1 << 20);
        buffer.Dispose();
        Action read = () => buffer.ToArray();
        Assert.Throws<ObjectDisposedException>(read);
        long before = GC.GetAllocatedBytesForCurrentThread();
        Assert.Throws<ObjectDisposedException>(read);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 65536, $"Rejected download allocated {allocated} bytes for a disposed 4 MiB buffer.");
    }
}
