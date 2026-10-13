using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class FinalLifecycleAuditTests(GpuFixture fixture)
{
    [Fact]
    public void RecursiveGraphCreationCannotReplaceTheRegisteredHostIdentity()
    {
        var original = new GraphTestNode(7);
        var adapter = new CallbackGraphAdapter();
        using var graph = fixture.Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>(
            [original, null], adapter, 2, 0);
        var view = graph.View;
        fixture.Gpu.For(0, 1, i => view.SetRoot(1, view.Create(new GraphTestRecord(11), 0)));

        Exception? recursiveError = null;
        adapter.CreateAction = () =>
        {
            adapter.CreateAction = null;
            recursiveError = Record.Exception(() => graph.CopyToHost());
        };
        var roots = graph.CopyToHost();
        var created = Assert.IsType<GraphTestNode>(roots[1]);
        Assert.Same(original, roots[0]);
        Assert.Equal(11, created.Number);
        Assert.Equal(graph.Download().Roots[1], graph.GetHandle(created));
        Assert.Same(created, graph.CopyToHost()[1]);
        Assert.Equal(1, adapter.CreatedCount);
        Assert.IsType<InvalidOperationException>(recursiveError);
    }

    [Fact]
    public void AbortedRecursiveGraphCreationCanBeRetriedWithStableIdentity()
    {
        var adapter = new CallbackGraphAdapter();
        using var graph = fixture.Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>(
            [null], adapter, 1, 0);
        var view = graph.View;
        fixture.Gpu.For(0, 1, i => view.SetRoot(0, view.Create(new GraphTestRecord(19), 0)));
        adapter.CreateAction = () =>
        {
            adapter.CreateAction = null;
            graph.CopyToHost();
        };

        Assert.Throws<InvalidOperationException>(() => graph.CopyToHost());
        Assert.Equal(0, adapter.WriteCount);
        Assert.Equal(0, adapter.CreatedCount);
        adapter.CreateAction = null;
        var created = Assert.IsType<GraphTestNode>(graph.CopyToHost()[0]);
        Assert.Equal(19, created.Number);
        Assert.Equal(graph.Download().Roots[0], graph.GetHandle(created));
        Assert.Same(created, graph.CopyToHost()[0]);
        Assert.Equal(1, adapter.CreatedCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WriteCallbacksRejectSameGraphReentryAndAllowReadsAndOtherGraphs(bool fromReferences)
    {
        var original = new GraphTestNode(7);
        var adapter = new CallbackGraphAdapter();
        using var graph = fixture.Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([original], adapter);
        var otherOriginal = new GraphTestNode(13);
        var otherAdapter = new GraphTestAdapter();
        using var other = fixture.Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>(
            [otherOriginal, null], otherAdapter, 2, 0);
        var view = graph.View;
        var otherView = other.View;
        fixture.Gpu.For(0, 1, i =>
        {
            view.Write(view.GetRoot(0), new GraphTestRecord(11));
            otherView.SetRoot(1, otherView.Create(new GraphTestRecord(17), 0));
        });

        int callbackCount = 0;
        Exception? recursiveError = null;
        GraphTestNode? otherCreated = null;
        void OnWrite()
        {
            adapter.WriteValueAction = null;
            adapter.WriteReferencesAction = null;
            callbackCount++;
            var snapshot = graph.Download();
            Assert.Equal(11, snapshot.Read(graph.GetHandle(original)).Number);
            recursiveError = Record.Exception(() => graph.CopyToHost());
            var otherRoots = other.CopyToHost();
            Assert.Same(otherOriginal, otherRoots[0]);
            otherCreated = Assert.IsType<GraphTestNode>(otherRoots[1]);
        }
        if (fromReferences)
            adapter.WriteReferencesAction = OnWrite;
        else
            adapter.WriteValueAction = OnWrite;

        Assert.Same(original, graph.CopyToHost()[0]);
        Assert.Equal(11, original.Number);
        Assert.Equal(1, callbackCount);
        Assert.IsType<InvalidOperationException>(recursiveError);
        Assert.Equal(1, adapter.WriteCount);
        Assert.Equal(1, adapter.ReferenceWriteCount);
        Assert.Equal(17, otherCreated!.Number);
        Assert.Equal(other.Download().Roots[1], other.GetHandle(otherCreated));
        Assert.Same(otherCreated, other.CopyToHost()[1]);
        Assert.Equal(1, otherAdapter.CreatedCount);
        Assert.Same(original, graph.CopyToHost()[0]);
    }

    private sealed class CallbackGraphAdapter : IGpuObjectGraphAdapter<GraphTestNode, GraphTestRecord>
    {
        public Action? CreateAction;
        public Action? WriteValueAction;
        public Action? WriteReferencesAction;
        public int CreatedCount;
        public int WriteCount;
        public int ReferenceWriteCount;

        public GraphTestRecord ReadValue(GraphTestNode source) => new(source.Number);
        public IReadOnlyList<GraphTestNode?> ReadReferences(GraphTestNode source) => source.References;

        public GraphTestNode CreateObject(GraphTestRecord value)
        {
            CreateAction?.Invoke();
            CreatedCount++;
            return new(value.Number);
        }

        public void WriteValue(GraphTestNode target, GraphTestRecord value)
        {
            WriteValueAction?.Invoke();
            WriteCount++;
            target.Number = value.Number;
        }

        public void WriteReferences(GraphTestNode target, IReadOnlyList<GraphTestNode?> references)
        {
            WriteReferencesAction?.Invoke();
            ReferenceWriteCount++;
            target.References = references.ToArray();
        }
    }
}
