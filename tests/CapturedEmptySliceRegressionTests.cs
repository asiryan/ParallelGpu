using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class CapturedEmptySliceRegressionTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void CapturedArraySegmentAtOwnerEndRetainsEmptyMetadata(int ownerLength)
    {
        int[] owner = new int[ownerLength];
        var segment = new ArraySegment<int>(owner, owner.Length, 0);
        int[] output = [-1, -1, -1];

        Gpu.For(0, 1, _ =>
        {
            output[0] = segment.Count;
            output[1] = segment.Slice(0, 0).Count;
            output[2] = segment.Slice(0).Count;
        });

        Assert.Equal(new[] { 0, 0, 0 }, output);
        Assert.Equal(new int[ownerLength], owner);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void CapturedMemoryAtOwnerEndRetainsEmptyMetadata(int ownerLength)
    {
        int[] owner = new int[ownerLength];
        Memory<int> memory = owner.AsMemory(owner.Length, 0);
        int[] output = [-1, -1, -1, -1];

        Gpu.For(0, 1, _ =>
        {
            output[0] = memory.Length;
            output[1] = memory.IsEmpty ? 1 : 0;
            output[2] = memory.Slice(0, 0).Span.Length;
            output[3] = memory.Slice(0).Length;
        });

        Assert.Equal(new[] { 0, 1, 0, 0 }, output);
        Assert.Equal(new int[ownerLength], owner);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void CapturedReadOnlyMemoryAtOwnerEndRetainsEmptyMetadata(int ownerLength)
    {
        int[] owner = new int[ownerLength];
        ReadOnlyMemory<int> memory = owner.AsMemory(owner.Length, 0);
        int[] output = [-1, -1, -1, -1];

        Gpu.For(0, 1, _ =>
        {
            output[0] = memory.Length;
            output[1] = memory.IsEmpty ? 1 : 0;
            output[2] = memory.Slice(0, 0).Span.Length;
            output[3] = memory.Slice(0).Length;
        });

        Assert.Equal(new[] { 0, 1, 0, 0 }, output);
        Assert.Equal(new int[ownerLength], owner);
    }

    [Theory]
    [InlineData("")]
    [InlineData("abc")]
    public void CapturedStringMemoryAtOwnerEndRetainsEmptyMetadata(string owner)
    {
        ReadOnlyMemory<char> memory = owner.AsMemory(owner.Length, 0);
        int[] output = [-1, -1, -1];

        Gpu.For(0, 1, _ =>
        {
            output[0] = memory.Length;
            output[1] = memory.IsEmpty ? 1 : 0;
            output[2] = memory.Slice(0, 0).Span.Length;
        });

        Assert.Equal(new[] { 0, 1, 0 }, output);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void CapturedAdapterAtOwnerEndRetainsEmptyMetadata(int ownerLength)
    {
        int[] owner = new int[ownerLength];
        var slice = new EmptyIndexedSlice(owner, owner.Length, 0);
        int[] output = [-1, -1];

        Gpu.For(0, 1, _ =>
        {
            output[0] = slice.Count;
            output[1] = slice.Length;
        });

        Assert.Equal(new[] { 0, 0 }, output);
        Assert.Equal(new int[ownerLength], owner);
    }

    [Fact]
    public void EmptyArraySegmentAccessThrowsAndRollsBackOtherCaptures()
    {
        int[] owner = [7, 11, 17];
        var segment = new ArraySegment<int>(owner, owner.Length, 0);
        int[] output = [41];

        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For(0, 1, _ =>
        {
            owner[0] = 99;
            output[0] = 99;
            segment[0] = 123;
        }));

        AssertRollbackAndRecovery(owner, output);
    }

    [Fact]
    public void EmptyMemoryAccessAndInvalidSliceThrowAndRollBackOtherCaptures()
    {
        int[] owner = [7, 11, 17];
        Memory<int> memory = owner.AsMemory(owner.Length, 0);
        int[] output = [41];

        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For(0, 1, _ =>
        {
            owner[0] = 99;
            output[0] = 99;
            memory.Span[0] = 123;
        }));
        Assert.Equal(new[] { 7, 11, 17 }, owner);
        Assert.Equal(new[] { 41 }, output);
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.For(0, 1, _ =>
        {
            owner[0] = 99;
            output[0] = memory.Slice(1).Length;
        }));

        AssertRollbackAndRecovery(owner, output);
    }

    [Fact]
    public void EmptyReadOnlyMemoryAccessThrowsAndRollsBackOtherCaptures()
    {
        int[] owner = [7, 11, 17];
        ReadOnlyMemory<int> memory = owner.AsMemory(owner.Length, 0);
        int[] output = [41];

        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For(0, 1, _ =>
        {
            owner[0] = 99;
            output[0] = memory.Slice(0, 0).Span[0];
        }));

        AssertRollbackAndRecovery(owner, output);
    }

    [Fact]
    public void EmptyAdapterAccessThrowsAndRollsBackOtherCaptures()
    {
        int[] owner = [7, 11, 17];
        var slice = new EmptyIndexedSlice(owner, owner.Length, 0);
        int[] output = [41];

        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For(0, 1, _ =>
        {
            owner[0] = 99;
            output[0] = 99;
            slice[0] = 123;
        }));

        AssertRollbackAndRecovery(owner, output);
    }

    [Fact]
    public void CachedLoopRefreshesEmptyAndNonemptyDescriptorsAndAliases()
    {
        int[] owner = [7, 11, 17];
        var segment = new ArraySegment<int>(owner, owner.Length, 0);
        Memory<int> memory = owner.AsMemory(owner.Length, 0);
        ReadOnlyMemory<int> readOnly = owner.AsMemory(owner.Length, 0);
        var slice = new EmptyIndexedSlice(owner, owner.Length, 0);
        int[] output = [-1, -1];
        Action<int> kernel = _ =>
        {
            output[0] = segment.Count + memory.Length + readOnly.Length + slice.Count;
            if (segment.Count > 0)
            {
                segment[0] += 1;
                memory.Span[0] += 2;
                slice[0] += 4;
                output[1] = readOnly.Span[0];
            }
            else
                output[1] = -1;
            owner[0] += 1;
        };

        Gpu.For(0, 1, kernel);
        Assert.Equal(new[] { 0, -1 }, output);
        Assert.Equal(new[] { 8, 11, 17 }, owner);

        segment = new(owner, 1, 1);
        memory = owner.AsMemory(1, 1);
        readOnly = owner.AsMemory(1, 1);
        slice = new(owner, 1, 1);
        Gpu.For(0, 1, kernel);
        Assert.Equal(new[] { 4, 18 }, output);
        Assert.Equal(new[] { 9, 18, 17 }, owner);

        segment = new(owner, owner.Length, 0);
        memory = owner.AsMemory(owner.Length, 0);
        readOnly = owner.AsMemory(owner.Length, 0);
        slice = new(owner, owner.Length, 0);
        Gpu.For(0, 1, kernel);
        Assert.Equal(new[] { 0, -1 }, output);
        Assert.Equal(new[] { 10, 18, 17 }, owner);
    }

    private void AssertRollbackAndRecovery(int[] owner, int[] output)
    {
        Assert.Equal(new[] { 7, 11, 17 }, owner);
        Assert.Equal(new[] { 41 }, output);
        Gpu.For(0, 1, _ => { owner[0] += 1; output[0] = owner[0]; });
        Assert.Equal(new[] { 8, 11, 17 }, owner);
        Assert.Equal(new[] { 8 }, output);
    }

    private sealed class EmptyIndexedSlice(int[] buffer, int offset, int count)
    {
        static EmptyIndexedSlice() => GpuCollectionAdapters.Register<EmptyIndexedSlice, int>(
            value => new(value.Buffer, value.Offset, value.Count));

        public int[] Buffer { get; } = buffer;
        public int Offset { get; } = offset;
        public int Count { get; } = count;
        public int Length => Count;
        public int this[int index]
        {
            get => Buffer[Offset + index];
            set => Buffer[Offset + index] = value;
        }
    }
}
