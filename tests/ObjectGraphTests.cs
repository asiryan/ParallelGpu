using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ObjectGraphTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    private static int ReadRootValue(GpuObjectGraphView<GraphTestRecord> view, int root) => view.Read(view.GetRoot(root)).Number;
    private static GpuObjectHandle CreateRecord(GpuObjectGraphView<GraphTestRecord> view, int number) => view.Create(new GraphTestRecord(number), 1);
    private static void LinkSelf(GpuObjectGraphView<GraphTestRecord> view, GpuObjectHandle node) => view.SetReference(node, 0, node);

    [Fact]
    public void NativeViewsAndHandlesCanBePassedThroughHelperMethods()
    {
        var original = new GraphTestNode(3);
        using var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([original], new GraphTestAdapter(), 2, 1);
        var view = graph.View;
        int[] result = [0];
        Gpu.For(0, 1, i =>
        {
            result[0] = ReadRootValue(view, 0);
            var node = CreateRecord(view, 8);
            LinkSelf(view, node);
            view.SetRoot(0, node);
        });
        Gpu.For(0, 1, i => result[0] += ReadRootValue(view, 0));
        Assert.Equal(11, result[0]);
        var root = Assert.IsType<GraphTestNode>(graph.CopyToHost()[0]);
        Assert.Equal(8, root.Number);
        Assert.Same(root, root.References[0]);
    }

    [Fact]
    public void SnapshotPreservesNullAliasesCyclesAndReferenceIdentity()
    {
        var first = new GraphTestNode(1);
        var second = new GraphTestNode(1);
        first.References = [first, second, null, second];
        second.References = [first];
        var adapter = new GraphTestAdapter();
        using var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([first, first, null, second], adapter);
        var firstHandle = graph.GetHandle(first);
        var secondHandle = graph.GetHandle(second);
        Assert.NotEqual(firstHandle, secondHandle);
        var initial = graph.Download();
        Assert.Equal(2, initial.Values.Count);
        Assert.Equal(new[] { firstHandle, firstHandle, default, secondHandle }, initial.Roots);
        Assert.Equal(new[] { firstHandle, secondHandle, default, secondHandle }, initial.GetReferences(firstHandle));
        Assert.Equal(firstHandle, initial.GetReferences(secondHandle)[0]);
        Assert.True(graph.GetHandle(null).IsNull);
        var view = graph.View;
        Gpu.For(0, 1, i => view.Write(firstHandle, new GraphTestRecord(42)));
        var updated = graph.Download();
        Assert.Equal(42, updated.Read(firstHandle).Number);
        Assert.Equal(1, first.Number);
        Assert.Equal(0, adapter.WriteCount);
        var roots = graph.CopyToHost();
        Assert.Same(first, roots[0]);
        Assert.Same(first, roots[1]);
        Assert.Null(roots[2]);
        Assert.Same(second, roots[3]);
        Assert.Equal(42, first.Number);
        Assert.Same(first, first.References[0]);
        Assert.Same(second, first.References[1]);
        Assert.Same(second, first.References[3]);
        Assert.Same(first, second.References[0]);
        graph.Dispose();
        Assert.Equal(42, updated.Read(firstHandle).Number);
    }

    [Fact]
    public void DeviceCreationAndRelinkingReconstructNewObjectsWithCycles()
    {
        var original = new GraphTestNode(7) { References = [null] };
        var adapter = new GraphTestAdapter();
        using var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([original, null], adapter, 3, 8);
        var view = graph.View;
        Gpu.For(0, 1, i =>
        {
            var parent = view.GetRoot(0);
            var child = view.Create(new GraphTestRecord(12), 3);
            view.SetReference(child, 0, parent);
            view.SetReference(child, 1, child);
            view.SetReference(parent, 0, child);
            view.SetRoot(1, child);
        });
        // A later kernel observes the fully initialized record and its references.
        Gpu.For(0, 1, i =>
        {
            var child = view.GetRoot(1);
            var value = view.Read(child);
            view.Write(child, new GraphTestRecord(value.Number + 1));
        });
        var snapshot = graph.Download();
        Assert.Equal(2, snapshot.Values.Count);
        Assert.Equal(13, snapshot.Read(snapshot.Roots[1]).Number);
        Assert.Null(original.References[0]);
        var roots = graph.CopyToHost();
        var created = Assert.IsType<GraphTestNode>(roots[1]);
        Assert.Same(original, roots[0]);
        Assert.Equal(13, created.Number);
        Assert.Same(created, original.References[0]);
        Assert.Same(original, created.References[0]);
        Assert.Same(created, created.References[1]);
        Assert.Null(created.References[2]);
        Assert.Equal(1, adapter.CreatedCount);
        Assert.Equal(snapshot.Roots[1], graph.GetHandle(created));
        Assert.Same(created, graph.CopyToHost()[1]);
        Assert.Equal(1, adapter.CreatedCount);
    }

    [Fact]
    public void ConcurrentRecordAndReferenceReservationsRemainDistinct()
    {
        const int count = 1031;
        var root = new GraphTestNode(-1);
        using var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([root], new GraphTestAdapter(), count + 1, count * 2);
        var view = graph.View;
        var handles = new GpuObjectHandle[count];
        Gpu.For(0, count, i =>
        {
            var node = view.Create(new GraphTestRecord(i), 2);
            view.SetReference(node, 0, node);
            view.SetReference(node, 1, default);
            handles[i] = node;
        });
        var snapshot = graph.Download();
        Assert.Equal(count + 1, snapshot.Values.Count);
        Assert.Equal(count, handles.Distinct().Count());
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(i, snapshot.Read(handles[i]).Number);
            Assert.Equal(handles[i], snapshot.GetReferences(handles[i])[0]);
            Assert.True(snapshot.GetReferences(handles[i])[1].IsNull);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AllocationOverflowRejectsBeforeHostReconstruction(bool exhaustReferences)
    {
        var original = new GraphTestNode(7);
        var adapter = new GraphTestAdapter();
        using var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([original], adapter,
            exhaustReferences ? 2 : 1, 0);
        var view = graph.View;
        int requestedReferences = exhaustReferences ? 1 : 0;
        Gpu.For(0, 1, i =>
        {
            view.Write(view.GetRoot(0), new GraphTestRecord(99));
            view.Create(new GraphTestRecord(12), requestedReferences);
        });
        Assert.Throws<InvalidOperationException>(() => graph.Download());
        Assert.Throws<InvalidOperationException>(() => graph.CopyToHost());
        Assert.Equal(7, original.Number);
        Assert.Equal(0, adapter.CreatedCount);
        Assert.Equal(0, adapter.WriteCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void InvalidGraphAccessNeverIndexesInvalidDeviceMemory(int mode)
    {
        var original = new GraphTestNode(7) { References = [null] };
        var adapter = new GraphTestAdapter();
        using var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([original], adapter, 2, 2);
        using var foreignGraph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([new GraphTestNode(8)], new GraphTestAdapter());
        var foreign = foreignGraph.Download().Roots[0];
        var view = graph.View;
        Gpu.For(0, 1, i =>
        {
            if (mode == 0)
                view.Read(default);
            if (mode == 1)
                view.Read(foreign);
            if (mode == 2)
                view.GetReference(view.GetRoot(0), 99);
            if (mode == 3)
                view.SetRoot(-1, view.GetRoot(0));
            if (mode == 4)
                view.Create(new GraphTestRecord(10), -1);
        });
        Assert.Throws<InvalidOperationException>(() => graph.Download());
        Assert.Throws<InvalidOperationException>(() => graph.CopyToHost());
        Assert.Equal(0, adapter.WriteCount);
        Assert.Equal(7, original.Number);
    }

    [Fact]
    public void EmptyGraphsHostArgumentsAndLifetimeAreValidated()
    {
        var adapter = new GraphTestAdapter();
        using var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([null, null], adapter);
        Assert.Empty(graph.Download().Values);
        Assert.Equal(new GraphTestNode?[] { null, null }, graph.CopyToHost());
        Assert.Throws<ArgumentException>(() => graph.GetHandle(new GraphTestNode(1)));
        Assert.Throws<ArgumentException>(() => graph.Download().Read(default));
        Assert.Throws<ArgumentNullException>(() => Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>(null!, adapter));
        Assert.Throws<ArgumentNullException>(() => Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([], null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([], adapter, -2));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([new GraphTestNode(1)], adapter, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([new GraphTestNode(1) { References = [null] }], adapter, 1, 0));
        graph.Dispose();
        Assert.Throws<ObjectDisposedException>(() => graph.View);
        Assert.Throws<ObjectDisposedException>(() => graph.Download());
        using var owner = new Gpu(new() { Backend = GpuBackend.Cpu });
        using var owned = owner.UploadObjectGraph<GraphTestNode, GraphTestRecord>([], adapter);
        owner.Dispose();
        Assert.Throws<ObjectDisposedException>(() => owned.CopyToHost());
    }

    [Fact]
    public void NativeGraphCaptureRejectsMappedBooleanRecordBeforeLaunchAndExecutorRecovers()
    {
        int[] markers = [0];
        var original = new GraphTestNode(3);
        var adapter = new BooleanGraphTestAdapter();
        Assert.Throws<NotSupportedException>(() => Gpu.UploadObjectGraph<GraphTestNode, BooleanGraphTestRecord>([original], adapter));
        GpuObjectGraphView<BooleanGraphTestRecord> mappedView = default;
        Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i =>
        {
            markers[i] = 99;
            mappedView.Write(mappedView.GetRoot(0), new BooleanGraphTestRecord(true, 8));
        }));
        Assert.Equal(0, markers[0]);
        Assert.Equal(3, original.Number);
        Assert.Equal(0, adapter.WriteCount);
        using var supported = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([original], new GraphTestAdapter());
        var nativeView = supported.View;
        Gpu.For(0, 1, i =>
        {
            markers[i] = 7;
            nativeView.Write(nativeView.GetRoot(0), new GraphTestRecord(8));
        });
        Assert.Equal(7, markers[0]);
        Assert.Equal(8, Assert.IsType<GraphTestNode>(supported.CopyToHost()[0]).Number);
    }

    [Fact]
    public void NativeGraphCapturesRejectDisposedDefaultAndForeignViewsBeforeEffects()
    {
        var markers = new int[1];
        var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([new GraphTestNode(7)], new GraphTestAdapter());
        var stale = graph.View;
        Action<int> body = i => markers[i] = stale.Read(stale.GetRoot(0)).Number;
        Gpu.For(0, 1, body);
        Assert.Equal(7, markers[0]);
        graph.Dispose();
        markers[0] = 0;
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, 1, body));
        Assert.Equal(0, markers[0]);
        using var foreignOwner = new Gpu(new() { Backend = GpuBackend.Cpu });
        using var foreignGraph = foreignOwner.UploadObjectGraph<GraphTestNode, GraphTestRecord>([new GraphTestNode(8)], new GraphTestAdapter());
        var foreign = foreignGraph.View;
        Assert.Throws<ArgumentException>(() => Gpu.For(0, 1, i => markers[i] = foreign.Read(foreign.GetRoot(0)).Number));
        Assert.Equal(0, markers[0]);
        GpuObjectGraphView<GraphTestRecord> empty = default;
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, 1, i => markers[i] = empty.Read(default).Number));
        Assert.Equal(0, markers[0]);
    }

    [Fact]
    public void TraversalDoesNotUseTheCallStackForDeepGraphs()
    {
        var root = new GraphTestNode(0);
        var tail = root;
        for (int i = 1; i < 2001; i++)
        {
            var next = new GraphTestNode(i);
            tail.References = [next];
            tail = next;
        }
        tail.References = [root];
        using var graph = Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([root], new GraphTestAdapter());
        Assert.Equal(2001, graph.Download().Values.Count);
        Assert.Same(root, graph.CopyToHost()[0]);
        Assert.Same(root, tail.References[0]);
    }
}

public readonly struct GraphTestRecord(int number)
{
    public readonly int Number = number;
}

public sealed class GraphTestNode(int number)
{
    public int Number = number;
    public GraphTestNode?[] References = [];
    public override bool Equals(object? obj) => obj is GraphTestNode other && Number == other.Number;
    public override int GetHashCode() => Number;
}

public sealed class GraphTestAdapter : IGpuObjectGraphAdapter<GraphTestNode, GraphTestRecord>
{
    public int CreatedCount { get; private set; }
    public int WriteCount { get; private set; }
    public GraphTestRecord ReadValue(GraphTestNode source) => new(source.Number);
    public IReadOnlyList<GraphTestNode?> ReadReferences(GraphTestNode source) => source.References;
    public GraphTestNode CreateObject(GraphTestRecord value)
    {
        CreatedCount++;
        return new(value.Number);
    }
    public void WriteValue(GraphTestNode target, GraphTestRecord value)
    {
        WriteCount++;
        target.Number = value.Number;
    }
    public void WriteReferences(GraphTestNode target, IReadOnlyList<GraphTestNode?> references) => target.References = references.ToArray();
}

public readonly struct BooleanGraphTestRecord(bool enabled, int number)
{
    public readonly bool Enabled = enabled;
    public readonly int Number = number;
}

public sealed class BooleanGraphTestAdapter : IGpuObjectGraphAdapter<GraphTestNode, BooleanGraphTestRecord>
{
    public int WriteCount { get; private set; }
    public BooleanGraphTestRecord ReadValue(GraphTestNode source) => new(true, source.Number);
    public IReadOnlyList<GraphTestNode?> ReadReferences(GraphTestNode source) => source.References;
    public GraphTestNode CreateObject(BooleanGraphTestRecord value) => new(value.Number);
    public void WriteValue(GraphTestNode target, BooleanGraphTestRecord value)
    {
        WriteCount++;
        target.Number = value.Number;
    }
    public void WriteReferences(GraphTestNode target, IReadOnlyList<GraphTestNode?> references) => target.References = references.ToArray();
}
