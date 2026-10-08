using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class FinalTranslatorAuditTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void DefaultSegmentIndexerPreservesClrExceptionAndRollsBackWrites()
    {
        ArraySegment<int> captured = default;
        Assert.Throws<ArgumentOutOfRangeException>(() => _ = captured[0]);
        int[] output = [41];
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = captured[0];
        }));
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Fact]
    public void LocalDefaultSegmentSlicePreservesClrExceptionAndRollsBackWrites()
    {
        Assert.Throws<InvalidOperationException>(() => _ = default(ArraySegment<int>).Slice(0));
        int[] output = [41];
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, _ =>
        {
            ArraySegment<int> local = default;
            output[0] = 99;
            output[0] = local.Slice(0).Count;
        }));
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Fact]
    public void NullSegmentConstructorPreservesClrExceptionAndRollsBackWrites()
    {
        int[]? missing = null;
        Assert.Throws<ArgumentNullException>(() => _ = new ArraySegment<int>(missing!, 0, 0));
        int[] output = [41];
        Assert.Throws<ArgumentNullException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            var local = new ArraySegment<int>(missing!, 0, 0);
            output[0] = local.Count;
        }));
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Fact]
    public void WholeSegmentConstructorMatchesRuntimeNullSemanticsAndAcceptsBackedEmptyArrays()
    {
        int[]? missing = null;
        int[] empty = [];
        int[] output = [41];
        var hostFailure = Record.Exception(() => _ = new ArraySegment<int>(missing!));
        Assert.Equal(0, new ArraySegment<int>(empty, 0, 0).Slice(0, 0).Count);
        Gpu.For<int>(0, 1, _ =>
        {
            var wholeEmpty = new ArraySegment<int>(empty);
            var backedEmpty = new ArraySegment<int>(empty, 0, 0);
            output[0] = wholeEmpty.Count + backedEmpty.Slice(0, 0).Count;
        });
        Assert.Equal(0, output[0]);
        var deviceFailure = Record.Exception(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            var segment = new ArraySegment<int>(missing!);
            output[0] = segment.Count;
        }));
        Assert.Equal(hostFailure?.GetType(), deviceFailure?.GetType());
        Assert.Equal(0, output[0]);
        Recover(output);
    }

    [Fact]
    public void SegmentSlicesRetainNullBackingAcrossCachedInvocations()
    {
        ArraySegment<int> segment = default;
        int[] output = [41];
        Action<int> kernel = _ => output[0] = segment.Slice(0, 0).Count;
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, kernel));
        Assert.Equal(41, output[0]);
        segment = new ArraySegment<int>(Array.Empty<int>());
        Gpu.For<int>(0, 1, kernel);
        Assert.Equal(0, output[0]);
        segment = default;
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, kernel));
        Assert.Equal(0, output[0]);
        Recover(output);
    }

    [Fact]
    public void SegmentIndexerFailureRollsBackBackingAndOtherCollections()
    {
        int[] data = [1, 2, 3, 4];
        var segment = new ArraySegment<int>(data, 1, 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => _ = segment[2]);
        int[] output = [41];
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            segment[0] = 99;
            segment[2] = 99;
        }));
        Assert.Equal(new[] { 1, 2, 3, 4 }, data);
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Fact]
    public void ListAndArrayInterfaceIndexersPreserveTheirExceptionTypes()
    {
        int[] data = [7];
        IList<int> listAlias = data;
        IReadOnlyList<int> readOnlyAlias = data;
        List<int> list = [7];
        Assert.Throws<ArgumentOutOfRangeException>(() => _ = listAlias[1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => _ = readOnlyAlias[1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => list[1] = 99);
        int[] output = [41];
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = listAlias[1];
        }));
        Assert.Equal(41, output[0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = readOnlyAlias[1];
        }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            list[1] = 99;
        }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            data[0] = 99;
            listAlias[1] = 99;
        }));
        Assert.Equal(7, data[0]);
        Assert.Equal(7, list[0]);
        Assert.Equal(41, output[0]);
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = listAlias[0];
            output[0] = data[1];
        }));
        Assert.Equal(41, output[0]);
        Recover(output);
    }

    [Fact]
    public void SegmentToMemoryAndSpanConversionsRestoreSpanBoundsSemantics()
    {
        int[] data = [1, 2, 3];
        var segment = new ArraySegment<int>(data, 1, 1);
        int[] output = [41];
        Assert.Throws<IndexOutOfRangeException>(() => _ = segment.AsSpan()[1]);
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            segment.AsSpan()[1] = 99;
        }));
        Assert.Equal(new[] { 1, 2, 3 }, data);
        Assert.Equal(41, output[0]);
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            ReadOnlyMemory<int> memory = segment.AsMemory();
            output[0] = memory.Span[1];
        }));
        Assert.Equal(41, output[0]);
        Recover(output);
    }
    [Fact]
    public void FrameworkIntegerClampSupportsValidInputsWithoutManagedTranslation()
    {
        int[] input = [-7, 3, 20];
        int[] output = new int[input.Length];
        var failure = Record.Exception(() => Gpu.For<int>(0, output.Length,
            i => output[i] = Math.Clamp(input[i], 0, 10)));
        if (failure is not null)
        {
            int[] recovery = [41];
            Recover(recovery);
        }
        Assert.Null(failure);
        Assert.Equal(new[] { 0, 3, 10 }, output);
    }
    [Fact]
    public void BoxedDefaultSegmentInterfaceIndexersRejectNullBackingBeforeBounds()
    {
        IList<int> writable = default(ArraySegment<int>);
        IReadOnlyList<int> readOnly = default(ArraySegment<int>);
        Assert.Throws<InvalidOperationException>(() => _ = writable[0]);
        Assert.Throws<InvalidOperationException>(() => writable[0] = 99);
        Assert.Throws<InvalidOperationException>(() => _ = readOnly[0]);
        int[] output = [41];
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = writable[0];
        }));
        Assert.Equal(41, output[0]);
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            writable[0] = 99;
        }));
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = readOnly[0];
        }));
        Assert.Equal(41, output[0]);
        Recover(output);
    }
    [Fact]
    public void DeferredInvalidReadCannotCreateArithmeticFailureOrPoisonFollowingLoops()
    {
        int[] empty = [];
        int[] output = [41];
        var failure = Record.Exception(() => Gpu.For<int>(0, 1, i =>
        {
            output[i] = 99;
            output[i] = 7 / empty[i];
        }));
        Assert.Equal(41, output[0]);
        Recover(output);
        Assert.IsType<IndexOutOfRangeException>(failure);
    }
    [Fact]
    public void NullArrayImplicitSegmentConversionKeepsDefaultValueSemantics()
    {
        int[]? missing = null;
        ArraySegment<int> host = missing!;
        Assert.Null(host.Array);
        Assert.Equal(0, host.AsSpan().Length);
        Assert.Equal(0, host.AsMemory().Length);
        int[] output = [41];
        Gpu.For<int>(0, 1, _ =>
        {
            ArraySegment<int> local = missing!;
            output[0] = local.Count + local.AsSpan().Length + local.AsMemory().Length;
        });
        Assert.Equal(0, output[0]);
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, _ =>
        {
            ArraySegment<int> local = missing!;
            output[0] = 99;
            output[0] = local.Slice(0).Count;
        }));
        Assert.Equal(0, output[0]);
        Recover(output);
    }
    private void Recover(int[] output)
    {
        Gpu.For<int>(0, output.Length, i => output[i] = i + 17);
        Assert.Equal(Enumerable.Range(0, output.Length).Select(i => i + 17), output);
    }
}
