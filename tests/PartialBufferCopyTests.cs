using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class PartialBufferCopyTests(GpuFixture fixture)
{
    [Fact]
    public void OffsetArrayCopiesPreserveElementsOutsideTheDeviceRange()
    {
        using var buffer = fixture.Gpu.Upload(new[] { 2, 3, 5, 7, 11, 13 });
        int[] source = [17, 19];
        buffer.CopyFrom(2, source);
        Assert.Equal(new[] { 2, 3, 17, 19, 11, 13 }, buffer.ToArray());
        Assert.Equal(new[] { 17, 19 }, source);

        int[] destination = new int[3];
        buffer.CopyTo(1, destination);
        Assert.Equal(new[] { 3, 17, 19 }, destination);
    }

    [Fact]
    public void ArrayRangeCopiesPreserveHostAndDeviceHeadsAndTails()
    {
        using var buffer = fixture.Gpu.Upload(new[] { 2, 3, 5, 7, 11, 13 });
        int[] source = [-1, 17, 19, -2];
        buffer.CopyFrom(2, source, 1, 2);
        int[] destination = [-3, -3, -3, -3, -3, -3, -3];
        buffer.CopyTo(1, destination, 2, 3);

        Assert.Equal(new[] { 2, 3, 17, 19, 11, 13 }, buffer.ToArray());
        Assert.Equal(new[] { -1, 17, 19, -2 }, source);
        Assert.Equal(new[] { -3, -3, 3, 17, 19, -3, -3 }, destination);
    }

    [Fact]
    public void SpanCopiesCompleteBeforeReturningAndKernelsSeeUploadedRanges()
    {
        using var buffer = fixture.Gpu.Upload(new[] { 2, 3, 5, 7, 11, 13 });
        Span<int> source = stackalloc int[] { 17, 19 };
        buffer.CopyFrom(2, source);
        source.Fill(-1);
        fixture.Gpu.ForEach(buffer, new TransformKernel());

        Span<int> destination = stackalloc int[] { -3, -3, -3, -3, -3, -3 };
        buffer.CopyTo(1, destination.Slice(1, 4));
        Assert.Equal(new[] { -3, 7, 36, 41, 26, -3 }, destination.ToArray());
        Assert.Equal(new[] { 4, 7, 36, 41, 26, 31 }, buffer.ToArray());
    }

    [Fact]
    public void PartialCopiesSupportUnmanagedStructElements()
    {
        var first = new CopyValue { Code = long.MinValue, Weight = -0.25 };
        var second = new CopyValue { Code = long.MaxValue, Weight = 1.5 };
        var replacement = new CopyValue { Code = 17, Weight = double.Epsilon };
        using var buffer = fixture.Gpu.Upload(new[] { first, second, first });
        buffer.CopyFrom(1, new[] { first, replacement, first }, 1, 1);
        var destination = new[] { second, second, second };
        buffer.CopyTo(1, destination, 1, 1);
        Assert.Equal(new[] { first, replacement, first }, buffer.ToArray());
        Assert.Equal(new[] { second, replacement, second }, destination);

        Span<CopyValue> stackDestination = stackalloc CopyValue[1];
        buffer.CopyTo(1, stackDestination);
        Assert.Equal(replacement, stackDestination[0]);
    }

    [Theory]
    [InlineData(-1, 0, 0, "destinationOffset", "sourceOffset")]
    [InlineData(int.MaxValue, 0, 0, "destinationOffset", "sourceOffset")]
    [InlineData(5, 0, 0, "destinationOffset", "sourceOffset")]
    [InlineData(0, -1, 0, "sourceOffset", "destinationOffset")]
    [InlineData(0, int.MaxValue, 0, "sourceOffset", "destinationOffset")]
    [InlineData(0, 4, 0, "sourceOffset", "destinationOffset")]
    [InlineData(0, 0, -1, "count", "count")]
    [InlineData(0, 0, int.MaxValue, "count", "count")]
    [InlineData(1, 1, int.MaxValue, "count", "count")]
    [InlineData(3, 0, 2, "count", "count")]
    [InlineData(0, 2, 2, "count", "count")]
    [InlineData(4, 0, 1, "count", "count")]
    [InlineData(0, 3, 1, "count", "count")]
    public void InvalidArrayRangesRejectWithoutCopyingOrOverflow(
        int deviceOffset, int hostOffset, int count, string uploadParameter, string downloadParameter)
    {
        using var buffer = fixture.Gpu.Upload(new[] { 2, 3, 5, 7 });
        int[] host = [17, 19, 23];
        Assert.Equal(uploadParameter, Assert.Throws<ArgumentOutOfRangeException>(
            () => buffer.CopyFrom(deviceOffset, host, hostOffset, count)).ParamName);
        Assert.Equal(downloadParameter, Assert.Throws<ArgumentOutOfRangeException>(
            () => buffer.CopyTo(deviceOffset, host, hostOffset, count)).ParamName);
        Assert.Equal(new[] { 2, 3, 5, 7 }, buffer.ToArray());
        Assert.Equal(new[] { 17, 19, 23 }, host);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(int.MaxValue)]
    [InlineData(3)]
    [InlineData(5)]
    public void OffsetArrayAndSpanCopiesRejectInvalidDeviceRanges(int offset)
    {
        using var buffer = fixture.Gpu.Upload(new[] { 2, 3, 5, 7 });
        int[] host = [17, 19];
        Assert.Equal("destinationOffset", Assert.Throws<ArgumentOutOfRangeException>(
            () => buffer.CopyFrom(offset, host)).ParamName);
        Assert.Equal("sourceOffset", Assert.Throws<ArgumentOutOfRangeException>(
            () => buffer.CopyTo(offset, host)).ParamName);
        Assert.Equal("destinationOffset", Assert.Throws<ArgumentOutOfRangeException>(
            () => buffer.CopyFrom(offset, (ReadOnlySpan<int>)host)).ParamName);
        Assert.Equal("sourceOffset", Assert.Throws<ArgumentOutOfRangeException>(
            () => buffer.CopyTo(offset, host.AsSpan())).ParamName);
        Assert.Equal(new[] { 2, 3, 5, 7 }, buffer.ToArray());
        Assert.Equal(new[] { 17, 19 }, host);
    }

    [Fact]
    public void EmptyCopiesAtBothRangeEndsAreValidAndLeaveContentsUntouched()
    {
        using var buffer = fixture.Gpu.Upload(new[] { 2, 3, 5, 7 });
        int[] host = [17, 19, 23];
        buffer.CopyFrom(buffer.Length, Array.Empty<int>());
        buffer.CopyTo(buffer.Length, Array.Empty<int>());
        buffer.CopyFrom(buffer.Length, ReadOnlySpan<int>.Empty);
        buffer.CopyTo(buffer.Length, Span<int>.Empty);
        buffer.CopyFrom(buffer.Length, host, host.Length, 0);
        buffer.CopyTo(buffer.Length, host, host.Length, 0);
        buffer.CopyFrom(0, host, host.Length, 0);
        buffer.CopyTo(0, host, host.Length, 0);
        Assert.Equal(new[] { 2, 3, 5, 7 }, buffer.ToArray());
        Assert.Equal(new[] { 17, 19, 23 }, host);

        using var empty = fixture.Gpu.Allocate<int>(0);
        empty.CopyFrom(Array.Empty<int>());
        empty.CopyTo(Array.Empty<int>());
        empty.CopyFrom(0, Array.Empty<int>());
        empty.CopyTo(0, Array.Empty<int>());
        empty.CopyFrom(0, ReadOnlySpan<int>.Empty);
        empty.CopyTo(0, Span<int>.Empty);
        empty.CopyFrom(0, host, host.Length, 0);
        empty.CopyTo(0, host, host.Length, 0);
        Assert.Empty(empty.ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => empty.CopyFrom(1, Array.Empty<int>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => empty.CopyTo(int.MaxValue, Array.Empty<int>()));
    }

    [Fact]
    public void ArrayOverloadsRejectNullBeforeInspectingRanges()
    {
        using var buffer = fixture.Gpu.Allocate<int>(2);
        Assert.Equal("source", Assert.Throws<ArgumentNullException>(() => buffer.CopyFrom(null!)).ParamName);
        Assert.Equal("destination", Assert.Throws<ArgumentNullException>(() => buffer.CopyTo(null!)).ParamName);
        Assert.Equal("source", Assert.Throws<ArgumentNullException>(() => buffer.CopyFrom(-1, (int[])null!)).ParamName);
        Assert.Equal("destination", Assert.Throws<ArgumentNullException>(() => buffer.CopyTo(-1, (int[])null!)).ParamName);
        Assert.Equal("source", Assert.Throws<ArgumentNullException>(() => buffer.CopyFrom(-1, null!, -1, -1)).ParamName);
        Assert.Equal("destination", Assert.Throws<ArgumentNullException>(() => buffer.CopyTo(-1, null!, -1, -1)).ParamName);
    }

    [Fact]
    public void LegacyWholeArrayCopiesStillRequireAnExactLength()
    {
        using var buffer = fixture.Gpu.Upload(new[] { 2, 3, 5, 7 });
        foreach (int length in new[] { 0, 3, 5 })
        {
            int[] host = Enumerable.Repeat(17, length).ToArray();
            Assert.Equal("source", Assert.Throws<ArgumentException>(() => buffer.CopyFrom(host)).ParamName);
            Assert.Equal("destination", Assert.Throws<ArgumentException>(() => buffer.CopyTo(host)).ParamName);
            Assert.Equal(Enumerable.Repeat(17, length), host);
        }
        Assert.Equal(new[] { 2, 3, 5, 7 }, buffer.ToArray());
        buffer.CopyFrom(new[] { 11, 13, 17, 19 });
        int[] destination = new int[4];
        buffer.CopyTo(destination);
        Assert.Equal(new[] { 11, 13, 17, 19 }, destination);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AllCopyOverloadsRejectDisposedBuffersAndOwnersIncludingEmptyRanges(bool disposeOwner)
    {
        using var gpu = new Gpu(new() { Backend = fixture.Gpu.Backend });
        using var buffer = gpu.Upload(new[] { 2, 3 });
        if (disposeOwner)
            gpu.Dispose();
        else
            buffer.Dispose();

        Assert.Throws<ObjectDisposedException>(() => buffer.CopyFrom(new[] { 17, 19 }));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyTo(new int[2]));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyFrom(2, Array.Empty<int>()));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyTo(2, Array.Empty<int>()));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyFrom(2, new[] { 17 }, 1, 0));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyTo(2, new[] { 17 }, 1, 0));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyFrom(2, ReadOnlySpan<int>.Empty));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyTo(2, Span<int>.Empty));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyFrom(-1, new[] { 17 }, -1, -1));
        Assert.Throws<ObjectDisposedException>(() => buffer.CopyTo(-1, new[] { 17 }, -1, -1));
    }

    public readonly struct TransformKernel : IGpuForEachKernel<int>
    {
        public void Execute(int index, ref int value) => value = 2 * value + index;
    }

    public struct CopyValue
    {
        public long Code;
        public double Weight;
    }
}
