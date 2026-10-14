using System.Buffers;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class MemoryManagerCommitRegressionTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeAndAdapterWritesShareFullStorageAndCommitEachCachedInvocation(bool nativeFirst)
    {
        ManagerSlice.Register();
        using var manager = new TestMemoryManager([7, 10, 20, 90]);
        Memory<int> native = manager.Memory.Slice(1, 2);
        var adapted = new ManagerSlice(manager, 1, 2);
        int[] output = [-1, -1];
        Action<int> body;
        if (nativeFirst)
            body = i =>
            {
                native.Span[i] += 3;
                adapted[i] += 5;
                output[i] = native.Span[i];
            };
        else
            body = i =>
            {
                adapted[i] += 5;
                native.Span[i] += 3;
                output[i] = adapted[i];
            };

        fixture.Gpu.For(0, 2, body);
        Assert.Equal(new[] { 7, 18, 28, 90 }, manager.GetSpan().ToArray());
        Assert.Equal(new[] { 18, 28 }, output);
        Assert.Equal(new[] { 7, 18, 28, 90 }, adapted.Committed);
        Assert.Equal(1, adapted.CommitCount);

        new[] { 8, 30, 40, 91 }.CopyTo(manager.GetSpan());
        fixture.Gpu.For(0, 2, body);
        Assert.Equal(new[] { 8, 38, 48, 91 }, manager.GetSpan().ToArray());
        Assert.Equal(new[] { 38, 48 }, output);
        Assert.Equal(new[] { 8, 38, 48, 91 }, adapted.Committed);
        Assert.Equal(2, adapted.CommitCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MultipleAdapterAliasesUseOneUserCommitForTheirManagerIdentity(bool nativeFirst)
    {
        ManagerSlice.Register();
        using var manager = new TestMemoryManager([1, 10, 20, 4]);
        Memory<int> native = manager.Memory.Slice(1, 2);
        var left = new ManagerSlice(manager, 1, 2);
        var right = new ManagerSlice(manager, 1, 2);
        var alias = left;
        int[] output = [-1, -1];
        if (nativeFirst)
            fixture.Gpu.For(0, 2, i =>
            {
                native.Span[i] += 1;
                left[i] += 2;
                right[i] += 4;
                output[i] = alias[i];
            });
        else
            fixture.Gpu.For(0, 2, i =>
            {
                left[i] += 2;
                right[i] += 4;
                native.Span[i] += 1;
                output[i] = alias[i];
            });

        Assert.Equal(new[] { 1, 17, 27, 4 }, manager.GetSpan().ToArray());
        Assert.Equal(new[] { 17, 27 }, output);
        Assert.Equal(1, left.CommitCount + right.CommitCount);
        Assert.Equal(new[] { 1, 17, 27, 4 }, left.Committed ?? right.Committed);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeMemoryUpdatesIndependentlyOfASideEffectOnlyUserCommit(bool nativeFirst)
    {
        ManagerSlice.Register();
        using var manager = new TestMemoryManager([7, 10, 20, 90]);
        Memory<int> native = manager.Memory.Slice(1, 2);
        var adapted = new ManagerSlice(manager, 1, 2, readOnly: true, copyOnCommit: false);
        int[] output = [-1, -1];
        if (nativeFirst)
            fixture.Gpu.For(0, 2, i =>
            {
                native.Span[i] += 3;
                output[i] = adapted[i];
            });
        else
            fixture.Gpu.For(0, 2, i =>
            {
                output[i] = adapted[i];
                native.Span[i] += 3;
                output[i] += 3;
            });

        Assert.Equal(new[] { 7, 13, 23, 90 }, manager.GetSpan().ToArray());
        Assert.Equal(new[] { 13, 23 }, output);
        Assert.Equal(new[] { 7, 13, 23, 90 }, adapted.Committed);
        Assert.Equal(adapted.Committed, adapted.MemoryAtCommit);
        Assert.Equal(1, adapted.CommitCount);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ReadingManagerStorageDoesNotInvokeUserCommit(bool nativeFirst, bool adapterReadOnly)
    {
        ManagerSlice.Register();
        using var manager = new TestMemoryManager([7, 10, 20, 90]);
        ReadOnlyMemory<int> native = manager.Memory.Slice(1, 2);
        var adapted = new ManagerSlice(manager, 1, 2, adapterReadOnly);
        int[] output = [-1, -1];
        if (nativeFirst)
            fixture.Gpu.For(0, 2, i => output[i] = native.Span[i] + adapted[i]);
        else
            fixture.Gpu.For(0, 2, i => output[i] = adapted[i] + native.Span[i]);

        Assert.Equal(new[] { 20, 40 }, output);
        Assert.Equal(new[] { 7, 10, 20, 90 }, manager.GetSpan().ToArray());
        Assert.Null(adapted.Committed);
        Assert.Equal(0, adapted.CommitCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeferredBoundsErrorSkipsManagerWritesAndUserCommitThenCachedBodyRecovers(bool nativeFirst)
    {
        ManagerSlice.Register();
        using var manager = new TestMemoryManager([7, 10, 20, 90]);
        Memory<int> native = manager.Memory.Slice(1, 2);
        var adapted = new ManagerSlice(manager, 1, 2);
        int[] output = [-1];
        int index = 2;
        Action<int> body;
        if (nativeFirst)
            body = _ =>
            {
                native.Span[0] += 3;
                adapted[0] += 5;
                output[0] = adapted[index];
            };
        else
            body = _ =>
            {
                adapted[0] += 5;
                native.Span[0] += 3;
                output[0] = native.Span[index];
            };

        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(new[] { 7, 10, 20, 90 }, manager.GetSpan().ToArray());
        Assert.Equal(new[] { -1 }, output);
        Assert.Null(adapted.Committed);
        Assert.Equal(0, adapted.CommitCount);

        index = 1;
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(new[] { 7, 18, 20, 90 }, manager.GetSpan().ToArray());
        Assert.Equal(new[] { 20 }, output);
        Assert.Equal(new[] { 7, 18, 20, 90 }, adapted.Committed);
        Assert.Equal(1, adapted.CommitCount);
    }

    private sealed class ManagerSlice(TestMemoryManager manager, int offset, int count,
        bool readOnly = false, bool copyOnCommit = true)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<ManagerSlice, int>(value =>
                new(value.manager.GetSpan().ToArray(), value.offset, value.Count,
                    value.readOnly, value.Commit, value.manager));
            return true;
        });
        private readonly TestMemoryManager manager = manager;
        private readonly int offset = offset;
        private readonly bool readOnly = readOnly;
        public static void Register() => _ = registered.Value;
        public int Count => count;
        public int CommitCount;
        public int[]? Committed;
        public int[]? MemoryAtCommit;
        private void Commit(int[] values)
        {
            CommitCount++;
            Committed = values.ToArray();
            MemoryAtCommit = manager.GetSpan().ToArray();
            if (copyOnCommit)
                values.CopyTo(manager.GetSpan());
        }
        public int this[int index]
        {
            get => manager.GetSpan()[offset + index];
            set => manager.GetSpan()[offset + index] = value;
        }
    }

    private sealed class TestMemoryManager(int[] values) : MemoryManager<int>
    {
        public override Span<int> GetSpan() => values;
        public override MemoryHandle Pin(int elementIndex = 0) => throw new NotSupportedException();
        public override void Unpin() { }
        protected override void Dispose(bool disposing) { }
    }
}
