using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class FinalCollectiveAuditTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CollectRejectsHiddenStorageBeforeCompilingTheOutputKernel(int kind)
    {
        switch (kind)
        {
            case 0: AssertRejected<AuditPublicInlineArray, FinalInlineCollectKernel>(); break;
            case 1: AssertRejected<AuditPublicFixedBuffer, FinalFixedCollectKernel>(); break;
            case 2: AssertRejected<AuditInlineContainer, FinalNestedInlineCollectKernel>(); break;
            default: AssertRejected<AuditFixedContainer, FinalNestedFixedCollectKernel>(); break;
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void EmptyAndReversedCollectRangesKeepTheirDocumentedSkipBehavior(int kind)
    {
        switch (kind)
        {
            case 0: AssertSkipped<AuditPublicInlineArray, FinalInlineCollectKernel>(); break;
            case 1: AssertSkipped<AuditPublicFixedBuffer, FinalFixedCollectKernel>(); break;
            case 2: AssertSkipped<AuditInlineContainer, FinalNestedInlineCollectKernel>(); break;
            default: AssertSkipped<AuditFixedContainer, FinalNestedFixedCollectKernel>(); break;
        }
    }

    private void AssertRejected<T, TKernel>() where T : unmanaged where TKernel : struct, IGpuCollectKernel<T>
    {
        // Validation happens before the launch even in the broken implementation;
        // these probes never run hidden-storage accesses on device storage.
        using var owner = new Gpu(new() { Backend = fixture.Gpu.Backend });
        var failure = Assert.Throws<NotSupportedException>(() => owner.Collect<T, TKernel>(0, 2, 2, new()));
        Assert.Contains("Inline arrays and fixed buffers", failure.Message);
        Assert.Equal(new[] { 0, 1 },
            owner.Collect<int, FinalCollectIndexesKernel>(0, 2, 2, new()).OrderBy(value => value));
    }

    private void AssertSkipped<T, TKernel>() where T : unmanaged where TKernel : struct, IGpuCollectKernel<T>
    {
        Assert.Empty(fixture.Gpu.Collect<T, TKernel>(0, 0, 0, new()));
        Assert.Empty(fixture.Gpu.Collect<T, TKernel>(1, -1, 0, new()));
    }
}

public readonly struct FinalInlineCollectKernel : IGpuCollectKernel<AuditPublicInlineArray>
{
    public void Execute(int index, GpuAppendView<AuditPublicInlineArray> output)
    {
        AuditPublicInlineArray value = default;
        value[1] = index;
        output.Add(value);
    }
}

public readonly struct FinalCollectIndexesKernel : IGpuCollectKernel<int>
{
    public void Execute(int index, GpuAppendView<int> output) => output.Add(index);
}

public readonly unsafe struct FinalFixedCollectKernel : IGpuCollectKernel<AuditPublicFixedBuffer>
{
    public void Execute(int index, GpuAppendView<AuditPublicFixedBuffer> output)
    {
        AuditPublicFixedBuffer value = default;
        value.Values[1] = index;
        output.Add(value);
    }
}

public readonly struct FinalNestedInlineCollectKernel : IGpuCollectKernel<AuditInlineContainer>
{
    public void Execute(int index, GpuAppendView<AuditInlineContainer> output)
    {
        AuditInlineContainer value = default;
        value.Payload[1] = index;
        output.Add(value);
    }
}

public readonly unsafe struct FinalNestedFixedCollectKernel : IGpuCollectKernel<AuditFixedContainer>
{
    public void Execute(int index, GpuAppendView<AuditFixedContainer> output)
    {
        AuditFixedContainer value = default;
        value.Payload.Values[1] = index;
        output.Add(value);
    }
}
