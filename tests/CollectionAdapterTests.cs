using System.Buffers;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class CollectionAdapterTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void ArraySegmentsAndArraysShareStorageAcrossSlices()
    {
        int[] values = [10, 20, 30, 40, 50];
        var segment = new ArraySegment<int>(values, 1, 3);
        int[] output = new int[3];
        Gpu.For<int>(0, segment.Count, i =>
        {
            segment[i] += 2;
            output[i] = values[i + 1];
        });
        Assert.Equal(new[] { 10, 22, 32, 42, 50 }, values);
        Assert.Equal(new[] { 22, 32, 42 }, output);
    }

    [Fact]
    public void MemoryReadOnlyMemoryAndSpanHelpersPreserveArrayAliases()
    {
        int[] values = [1, 2, 3, 4, 5];
        Memory<int> mutable = values.AsMemory(1, 3);
        ReadOnlyMemory<int> readOnly = values.AsMemory(1, 3);
        int[] output = new int[3];
        Gpu.For<int>(0, mutable.Length, i =>
        {
            mutable.Span[i] += 10;
            output[i] = Read(readOnly.Span, i);
        });
        Assert.Equal(new[] { 1, 12, 13, 14, 5 }, values);
        Assert.Equal(new[] { 12, 13, 14 }, output);
    }

    [Fact]
    public void LocalMemoryConstructorsAndSlicesHaveZeroBasedIndices()
    {
        int[] values = [1, 2, 3, 4, 5];
        int[] output = [-1, -1];
        Gpu.For<int>(0, output.Length, i =>
        {
            var memory = new Memory<int>(values, 1, 3);
            Span<int> tail = memory.Slice(1).Span;
            tail[i] *= 2;
            output[i] = values.AsSpan(2, 2)[i];
        });
        Assert.Equal(new[] { 1, 2, 6, 8, 5 }, values);
        Assert.Equal(new[] { 6, 8 }, output);
    }

    [Fact]
    public void NullableMemoryAndDefaultEmptyViewsAreSupported()
    {
        int?[] values = [null, 2, 3];
        Memory<int?> memory = values;
        Memory<int> empty = default;
        bool[] result = [false];
        Gpu.For<int?>(0, values.Length, i => memory.Span[i] += 4);
        Assert.Equal(new int?[] { null, 6, 7 }, values);
        Gpu.For(0, 1, i => result[i] = empty.IsEmpty && empty.Span.Length == 0);
        Assert.Equal(new[] { true }, result);
    }

    [Fact]
    public void StringBackedReadOnlyMemoryRetainsFullUtf16Characters()
    {
        ReadOnlyMemory<char> input = "a\uFFFF\u8000z".AsMemory(1, 2);
        char[] output = new char[2];
        Gpu.For<char>(0, input.Length, i => output[i] = input.Span[i]);
        Assert.Equal(new[] { '\uFFFF', '\u8000' }, output);
    }

    [Fact]
    public void MemoryManagerAliasesShareOneSnapshotAndCommit()
    {
        using var manager = new TestMemoryManager([1, 2, 3, 4]);
        Memory<int> mutable = manager.Memory.Slice(1, 2);
        ReadOnlyMemory<int> readOnly = manager.Memory.Slice(1, 2);
        int[] output = new int[2];
        Gpu.For<int>(0, mutable.Length, i =>
        {
            mutable.Span[i] += 5;
            output[i] = readOnly.Span[i];
        });
        Assert.Equal(new[] { 1, 7, 8, 4 }, manager.GetSpan().ToArray());
        Assert.Equal(new[] { 7, 8 }, output);
    }

    [Fact]
    public void RegisteredCollectionsRunOnlySnapshotCallbacksAndPreserveAliases()
    {
        int[] backing = [1, 2, 3, 4];
        var left = new IndexedSlice(backing, 0, 2);
        var right = new IndexedSlice(backing, 0, 2);
        GpuCollectionAdapters.Register<IndexedSlice, int>(collection =>
            new(collection.Buffer, collection.Offset, collection.Count));
        int[] output = [-1, -1];
        Gpu.For<int>(0, left.Count, i =>
        {
            left[i] += 10;
            output[i] = right[i];
        });
        Assert.Equal(new[] { 11, 12, 3, 4 }, backing);
        Assert.Equal(new[] { 11, 12 }, output);
        Assert.Equal(0, left.IndexerCalls);
        Assert.Equal(0, right.IndexerCalls);
        Assert.Throws<InvalidOperationException>(() =>
            GpuCollectionAdapters.Register<IndexedSlice, int>(collection => new(collection.Buffer)));
    }

    [Fact]
    public void StagedAdaptersShareAnIdentityAndCommitOnlyAfterSuccess()
    {
        int[] backing = [1, 2];
        var left = new StagedSlice(backing);
        var right = new StagedSlice(backing);
        int commits = 0;
        GpuCollectionAdapters.Register<StagedSlice, int>(collection =>
            new(collection.Buffer.ToArray(), commit: values => { values.CopyTo(collection.Buffer, 0); commits++; }, identity: collection.Buffer));
        int[] output = [-1, -1];
        Gpu.For<int>(0, left.Count, i => { left[i] += 4; output[i] = right[i]; });
        Assert.Equal(new[] { 5, 6 }, backing);
        Assert.Equal(new[] { 5, 6 }, output);
        Assert.Equal(1, commits);
        Assert.Throws<IndexOutOfRangeException>(() =>
            Gpu.For<int>(0, 1, i => { left[i] += 10; output[i] = right[99]; }));
        Assert.Equal(new[] { 5, 6 }, backing);
        Assert.Equal(1, commits);
    }

    [Fact]
    public void AdapterReadOnlyViewsRejectWritesBeforeHostChanges()
    {
        int[] backing = [1, 2];
        var input = new ReadOnlyIndexedSlice(backing);
        GpuCollectionAdapters.Register<ReadOnlyIndexedSlice, int>(collection => new(collection.Buffer, true));
        int[] output = [-1, -1];
        Assert.Throws<InvalidOperationException>(() =>
            Gpu.For<int>(0, backing.Length, i => { output[i] = input[i]; input[i] = 9; }));
        Assert.Equal(new[] { 1, 2 }, backing);
        Assert.Equal(new[] { -1, -1 }, output);
    }

    [Fact]
    public void InvalidSpanIndicesAndSlicesLeaveHostStorageUnchanged()
    {
        int[] values = [1, 2];
        Memory<int> memory = values;
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, i => memory.Span[3] = 9));
        Assert.Equal(new[] { 1, 2 }, values);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Gpu.For<int>(0, 1, i => memory.Slice(1, 2).Span[0] = 9));
        Assert.Equal(new[] { 1, 2 }, values);
    }

    [Fact]
    public void DefaultSpanLocalsRetainDeferredErrorContext()
    {
        int[] output = [-1];
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, i =>
        {
            Span<int> empty = default;
            output[i] = empty[0];
        }));
        Assert.Equal(new[] { -1 }, output);
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For<int>(0, 1, i =>
        {
            ReadOnlySpan<int> empty = default;
            output[i] = empty.Slice(1).Length;
        }));
        Assert.Equal(new[] { -1 }, output);
    }

    [Fact]
    public void RegisteredInterfaceAdaptersResolveRuntimeImplementations()
    {
        IIndexed collection = new InterfaceIndexed([1, 2]);
        GpuCollectionAdapters.Register<IIndexed, int>(value => new(value.Buffer));
        int[] output = [-1, -1];
        Gpu.For<int>(0, output.Length, i => { collection[i] += 3; output[i] = collection[i]; });
        Assert.Equal(new[] { 4, 5 }, collection.Buffer);
        Assert.Equal(new[] { 4, 5 }, output);
    }

    [Fact]
    public void UnrelatedRuntimeAdaptersAreRejectedWithAnExplicitDiagnostic()
    {
        IAmbiguousA collection = new AmbiguousIndexed([1]);
        GpuCollectionAdapters.Register<IAmbiguousA, int>(value => new(value.Buffer));
        GpuCollectionAdapters.Register<IAmbiguousB, int>(value => new(value.Buffer));
        int[] output = [-1];
        var error = Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, i => output[i] = collection[i]));
        Assert.Contains("several unrelated", error.Message);
        Assert.Equal(new[] { -1 }, output);
    }

    [Fact]
    public void AdapterRegistrationAfterValueCompilationIsRejected()
    {
        FrozenIndexedValue value = new() { Number = 7 };
        int[] output = [-1];
        Gpu.For<int>(0, 1, i => output[i] = value.Number);
        Assert.Equal(new[] { 7 }, output);
        var error = Assert.Throws<InvalidOperationException>(() =>
            GpuCollectionAdapters.Register<FrozenIndexedValue, int>(item => new([item.Number])));
        Assert.Contains("before compiling", error.Message);
    }

    [Fact]
    public void NullReferenceAdaptersKeepNullReferenceSemantics()
    {
        GpuCollectionAdapters.Register<NullIndexed, int>(value => new(value.Buffer));
        NullIndexed? input = null;
        int[] output = [-1];
        Gpu.For<int>(0, 1, i => output[i] = input is null ? 9 : input[i]);
        Assert.Equal(new[] { 9 }, output);
        Assert.Throws<NullReferenceException>(() => Gpu.For<int>(0, 1, i => output[i] = input![i]));
        Assert.Equal(new[] { 9 }, output);
    }
    [Fact]
    public void UnsupportedRangeSlicesFailBeforeCompilationAndDoNotPoisonLaterLoops()
    {
        int[] values = [1, 2, 3];
        int[] output = [-1];
        Range range = 1..3;
        Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, i => output[i] = values.AsSpan(range)[i]));
        Assert.Equal(new[] { -1 }, output);
        Gpu.For<int>(0, 1, i => output[i] = values.AsSpan(1, 2)[i]);
        Assert.Equal(new[] { 2 }, output);
    }
    [Fact]
    public void StagedArrayIdentitiesAliasIndependentNativeArrayCaptures()
    {
        int[] backing = [1, 2, 3];
        var view = new StagedArrayView(backing);
        int commits = 0;
        GpuCollectionAdapters.Register<StagedArrayView, int>(collection =>
            new(collection.Buffer.ToArray(), commit: values => { values.CopyTo(collection.Buffer, 0); commits++; }, identity: collection.Buffer));
        int[] output = [-1, -1, -1];
        Gpu.For<int>(0, output.Length, i => { view[i] += 10; output[i] = backing[i]; });
        Assert.Equal(new[] { 11, 12, 13 }, backing);
        Assert.Equal(backing, output);
        Assert.Equal(1, commits);
        Gpu.For<int>(0, output.Length, i => { backing[i] += 2; output[i] = view[i]; });
        Assert.Equal(new[] { 13, 14, 15 }, backing);
        Assert.Equal(backing, output);
        Assert.Equal(2, commits);
    }
    private static int Read(ReadOnlySpan<int> values, int index) => values[index];

    private sealed class IndexedSlice(int[] buffer, int offset, int count)
    {
        public int[] Buffer { get; } = buffer;
        public int Offset { get; } = offset;
        public int Count { get; } = count;
        public int IndexerCalls { get; private set; }
        public int this[int index]
        {
            get { IndexerCalls++; return Buffer[Offset + index]; }
            set { IndexerCalls++; Buffer[Offset + index] = value; }
        }
    }

    private sealed class StagedSlice(int[] buffer)
    {
        public int[] Buffer { get; } = buffer;
        public int Count => Buffer.Length;
        public int this[int index] { get => Buffer[index]; set => Buffer[index] = value; }
    }

    private sealed class ReadOnlyIndexedSlice(int[] buffer)
    {
        public int[] Buffer { get; } = buffer;
        public int Count => Buffer.Length;
        public int this[int index] { get => Buffer[index]; set => Buffer[index] = value; }
    }

    private interface IIndexed
    {
        int[] Buffer { get; }
        int this[int index] { get; set; }
    }
    private sealed class InterfaceIndexed(int[] buffer) : IIndexed
    {
        public int[] Buffer { get; } = buffer;
        public int this[int index] { get => Buffer[index]; set => Buffer[index] = value; }
    }
    private interface IAmbiguousA { int[] Buffer { get; } int this[int index] { get; } }
    private interface IAmbiguousB { int[] Buffer { get; } int this[int index] { get; } }
    private sealed class AmbiguousIndexed(int[] buffer) : IAmbiguousA, IAmbiguousB
    {
        public int[] Buffer { get; } = buffer;
        public int this[int index] => Buffer[index];
    }
    private struct FrozenIndexedValue
    {
        public int Number;
        public int this[int index] => Number;
    }
    private sealed class NullIndexed(int[] buffer)
    {
        public int[] Buffer { get; } = buffer;
        public int this[int index] => Buffer[index];
    }
    private sealed class StagedArrayView(int[] buffer)
    {
        public int[] Buffer { get; } = buffer;
        public int this[int index] { get => Buffer[index]; set => Buffer[index] = value; }
    }
    private sealed class TestMemoryManager(int[] values) : MemoryManager<int>
    {
        public override Span<int> GetSpan() => values;
        public override MemoryHandle Pin(int elementIndex = 0) => throw new NotSupportedException();
        public override void Unpin() { }
        protected override void Dispose(bool disposing) { }
    }
}
