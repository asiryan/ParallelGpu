using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ResourceAuditRegressionTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void EmptyReductionRejectsHiddenStorageBeforeEvaluatingTheIdentity(int kind)
    {
        switch (kind)
        {
            case 0: AssertRejected<AuditPublicInlineArray>(); break;
            case 1: AssertRejected<AuditPublicFixedBuffer>(); break;
            case 2: AssertRejected<AuditInlineContainer>(); break;
            default: AssertRejected<AuditFixedContainer>(); break;
        }

        Assert.Equal(0, fixture.Gpu.Reduce(Array.Empty<int>(), new Int32Sum()));
    }

    private void AssertRejected<T>() where T : unmanaged
    {
        var failure = Assert.Throws<NotSupportedException>(() =>
            fixture.Gpu.Reduce(Array.Empty<T>(), new ResourceAuditRejectIdentity<T>()));
        Assert.Contains("Inline arrays and fixed buffers", failure.Message);
    }

    [Fact]
    public void GraphWriteCallbackFailureKeepsNewObjectIdentityForRetry()
    {
        var original = new GraphTestNode(7);
        var adapter = new RetryGraphAdapter { FailNextWrite = true };
        using var graph = fixture.Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([original], adapter, 2, 0);
        var view = graph.View;
        fixture.Gpu.For(0, 1, _ => view.SetRoot(0, view.Create(new GraphTestRecord(12), 0)));

        var failure = Assert.Throws<InvalidOperationException>(() => graph.CopyToHost());
        Assert.Equal("Intentional graph write failure.", failure.Message);
        Assert.Equal(1, adapter.CreatedCount);
        Assert.Equal(7, original.Number);
        var created = Assert.IsType<GraphTestNode>(adapter.Created);

        var roots = graph.CopyToHost();
        Assert.Same(created, roots[0]);
        Assert.Equal(12, created.Number);
        Assert.Equal(1, adapter.CreatedCount);
        Assert.Equal(graph.Download().Roots[0], graph.GetHandle(created));
    }

    [Fact]
    public void DuplicateGraphFactoryResultsRejectBeforeHostWritesAndAllowRetry()
    {
        var original = new GraphTestNode(7);
        var adapter = new RetryGraphAdapter { Duplicate = original };
        using var graph = fixture.Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>([original], adapter, 2, 0);
        var view = graph.View;
        fixture.Gpu.For(0, 1, _ =>
        {
            view.Write(view.GetRoot(0), new GraphTestRecord(99));
            view.SetRoot(0, view.Create(new GraphTestRecord(12), 0));
        });

        var failure = Assert.Throws<InvalidOperationException>(() => graph.CopyToHost());
        Assert.Contains("distinct non-null", failure.Message);
        Assert.Equal(7, original.Number);
        Assert.Equal(0, adapter.Writes);

        adapter.Duplicate = null;
        var roots = graph.CopyToHost();
        Assert.Same(adapter.Created, roots[0]);
        Assert.Equal(12, Assert.IsType<GraphTestNode>(roots[0]).Number);
        Assert.Equal(99, original.Number);
    }

    private sealed class RetryGraphAdapter : IGpuObjectGraphAdapter<GraphTestNode, GraphTestRecord>
    {
        public bool FailNextWrite;
        public GraphTestNode? Duplicate;
        public GraphTestNode? Created;
        public int CreatedCount;
        public int Writes;
        public GraphTestRecord ReadValue(GraphTestNode source) => new(source.Number);
        public IReadOnlyList<GraphTestNode?> ReadReferences(GraphTestNode source) => source.References;
        public GraphTestNode CreateObject(GraphTestRecord value)
        {
            CreatedCount++;
            return Duplicate ?? (Created = new GraphTestNode(value.Number));
        }
        public void WriteValue(GraphTestNode target, GraphTestRecord value)
        {
            if (FailNextWrite)
            {
                FailNextWrite = false;
                throw new InvalidOperationException("Intentional graph write failure.");
            }
            Writes++;
            target.Number = value.Number;
        }
        public void WriteReferences(GraphTestNode target, IReadOnlyList<GraphTestNode?> references) =>
            target.References = references.ToArray();
    }
}

public readonly struct ResourceAuditRejectIdentity<T> : IGpuReduction<T> where T : unmanaged
{
    public T Identity => throw new InvalidOperationException("Unsupported storage must be rejected before the host identity runs.");
    public T Combine(T left, T right) => default;
}
