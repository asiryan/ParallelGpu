using System.Buffers;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class FinalTransferAuditTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void DistinctAdapterOwnersSharingBackingStorageEachReceiveTheirCommit()
    {
        int[] backing = [1, 2];
        var first = new SharedOwner(backing);
        var second = new SharedOwner(backing);
        GpuCollectionAdapters.Register<SharedOwner, int>(owner =>
            new(owner.Buffer, commit: values => owner.CommitCount++, identity: owner));
        int[] observed = [-1, -1];
        Gpu.For<int>(0, backing.Length, i =>
        {
            first[i] += 5;
            observed[i] = second[i];
        });
        Assert.Equal(new[] { 6, 7 }, backing);
        Assert.Equal(backing, observed);
        Assert.Equal(1, first.CommitCount);
        Assert.Equal(1, second.CommitCount);
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, i =>
        {
            first[i] += 99;
            observed[i] = second[99];
        }));
        Assert.Equal(new[] { 6, 7 }, backing);
        Assert.Equal(backing, observed);
        Assert.Equal(1, first.CommitCount);
        Assert.Equal(1, second.CommitCount);
    }

    [Fact]
    public void MemoryOwnerIdentitiesRejectInconsistentFullStorageSnapshots()
    {
        using var manager = new AuditMemoryManager([10]);
        var alias = new ManagerAlias(manager, [7, 8]);
        GpuCollectionAdapters.Register<ManagerAlias, int>(owner =>
            new(owner.Buffer, identity: owner.Identity));
        Memory<int> memory = manager.Memory;
        int[] output = [-1];
        var error = Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1,
            i => output[i] = alias[i] + memory.Span[i]));
        Assert.Contains("full buffer length", error.Message);
        Assert.Equal(new[] { -1 }, output);
        Assert.Equal(new[] { 10 }, manager.GetSpan().ToArray());
    }

    [Fact]
    public void CapturedDefaultArraySegmentsRetainNullBackingSemantics()
    {
        ArraySegment<int> empty = default;
        var backedEmpty = new ArraySegment<int>(Array.Empty<int>());
        int[] output = [-1];
        Gpu.For<int>(0, 1, i => output[i] = empty.Count + backedEmpty.Slice(0, 0).Count);
        Assert.Equal(new[] { 0 }, output);
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1,
            i => output[i] = empty.Slice(0, 0).Count));
        Assert.Equal(new[] { 0 }, output);
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For<int>(0, 1,
            i => output[i] = empty[i]));
        Assert.Equal(new[] { 0 }, output);
    }

    [Fact]
    public void MemoryOwnerAliasesKeepTheBuiltInCommitWhenCustomSnapshotHasNone()
    {
        using var manager = new AuditMemoryManager([10, 20]);
        var alias = new CommitlessManagerAlias(manager, manager.GetSpan().ToArray());
        GpuCollectionAdapters.Register<CommitlessManagerAlias, int>(owner =>
            new(owner.Buffer, identity: owner.Identity));
        Memory<int> memory = manager.Memory;
        int[] output = [-1, -1];
        Gpu.For<int>(0, output.Length, i =>
        {
            int previous = alias[i];
            memory.Span[i] = previous + 3;
            output[i] = alias[i];
        });
        Assert.Equal(new[] { 13, 23 }, manager.GetSpan().ToArray());
        Assert.Equal(new[] { 13, 23 }, output);
    }

    private sealed class CommitlessManagerAlias(object identity, int[] buffer)
    {
        public object Identity { get; } = identity;
        public int[] Buffer { get; } = buffer;
        public int this[int index] => Buffer[index];
    }

    private sealed class ManagerAlias(object identity, int[] buffer)
    {
        public object Identity { get; } = identity;
        public int[] Buffer { get; } = buffer;
        public int this[int index] => Buffer[index];
    }

    private sealed class AuditMemoryManager(int[] values) : MemoryManager<int>
    {
        public override Span<int> GetSpan() => values;
        public override MemoryHandle Pin(int elementIndex = 0) => throw new NotSupportedException();
        public override void Unpin() { }
        protected override void Dispose(bool disposing) { }
    }

    private sealed class SharedOwner(int[] buffer)
    {
        public int[] Buffer { get; } = buffer;
        public int CommitCount;
        public int this[int index] { get => Buffer[index]; set => Buffer[index] = value; }
    }
}
