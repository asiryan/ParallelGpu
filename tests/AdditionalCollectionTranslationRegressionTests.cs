using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AdditionalCollectionTranslationRegressionTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DefaultViewsReturnSafeWideReferencesThroughHelpers(bool readOnly)
    {
        int[] output = [41];
        Action<int> body = _ =>
        {
            output[0] = 99;
            if (readOnly)
            {
                ReadOnlyMemory<WideValue> memory = default;
                output[0] = (int)ReadOnlyElement(in memory, 0).Last;
            }
            else
            {
                Memory<WideValue> memory = default;
                ref WideValue element = ref MutableElement(ref memory, 0);
                element = new WideValue(long.MaxValue);
                output[0] = (int)element.Last;
            }
        };
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(new[] { 41 }, output);
        fixture.Gpu.For(0, 1, i => output[i] = 17);
        Assert.Equal(new[] { 17 }, output);
    }

    [Fact]
    public void DefaultReadOnlySpanKeepsContextAcrossSliceAndRefReturningHelper()
    {
        int[] output = [41];
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            ReadOnlySpan<WideValue> empty = default;
            var sliced = empty.Slice(0, 0);
            output[0] = 99;
            output[0] = (int)ReadSpan(sliced, 0).Last;
        }));
        Assert.Equal(new[] { 41 }, output);
        fixture.Gpu.For(0, 1, i => output[i] = 23);
        Assert.Equal(new[] { 23 }, output);
    }

    [Fact]
    public void RefHelperCanReplaceLocalMemorySliceWithoutChangingCapturedDescriptor()
    {
        int[] owner = [10, 20, 30, 40];
        Memory<int> captured = owner;
        int[] output = [-1, -1, -1];
        fixture.Gpu.For(0, output.Length, i =>
        {
            var local = captured;
            Trim(ref local);
            local.Span[i] += 5;
            output[i] = captured.Length + local.Length + local.Span[i];
        });
        Assert.Equal(new[] { 10, 25, 35, 45 }, owner);
        Assert.Equal(new[] { 32, 42, 52 }, output);
        Assert.Equal(4, captured.Length);
        Assert.Equal(10, captured.Span[0]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NullMemoryConstructorsPreserveDefaultValueAndSpanBoundsSemantics(bool readOnly)
    {
        int[]? absent = null;
        int[] output = [41];
        fixture.Gpu.For(0, 1, _ => output[0] = readOnly
            ? MakeReadOnly(absent!, 0, 0).Slice(0, 0).Span.Length
            : MakeMutable(absent!, 0, 0).Slice(0, 0).Span.Length);
        Assert.Equal(new[] { 0 }, output);
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = readOnly ? MakeReadOnly(absent!, 0, 0).Span[0]
                : MakeMutable(absent!, 0, 0).Span[0];
        }));
        Assert.Equal(new[] { 0 }, output);
    }

    [Theory]
    [InlineData(-1, 0)]
    [InlineData(0, -1)]
    [InlineData(1, 0)]
    [InlineData(0, 1)]
    [InlineData(int.MaxValue, int.MaxValue)]
    public void InvalidNullMemoryConstructorRangesMatchClrExceptionTypes(int start, int length)
    {
        int[]? absent = null;
        int[] output = [41];
        var mutableHost = Record.Exception(() => _ = MakeMutable(absent!, start, length));
        var mutableDevice = Record.Exception(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = MakeMutable(absent!, start, length).Length;
        }));
        Assert.IsType<ArgumentOutOfRangeException>(mutableHost);
        Assert.Equal(mutableHost!.GetType(), mutableDevice?.GetType());
        Assert.Equal(new[] { 41 }, output);

        var readOnlyHost = Record.Exception(() => _ = MakeReadOnly(absent!, start, length));
        var readOnlyDevice = Record.Exception(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = MakeReadOnly(absent!, start, length).Length;
        }));
        Assert.IsType<ArgumentOutOfRangeException>(readOnlyHost);
        Assert.Equal(readOnlyHost!.GetType(), readOnlyDevice?.GetType());
        Assert.Equal(new[] { 41 }, output);
    }

    private static Memory<int> MakeMutable(int[] values, int start, int length) => new(values, start, length);
    private static ReadOnlyMemory<int> MakeReadOnly(int[] values, int start, int length) => new(values, start, length);
    private static void Trim(ref Memory<int> values) => values = values.Slice(1);
    private static ref WideValue MutableElement(ref Memory<WideValue> values, int index) => ref values.Span[index];
    private static ref readonly WideValue ReadOnlyElement(in ReadOnlyMemory<WideValue> values, int index) => ref values.Span[index];
    private static ref readonly WideValue ReadSpan(ReadOnlySpan<WideValue> values, int index) => ref values[index];

    private struct WideValue
    {
        public long First, Second, Third, Fourth, Fifth, Sixth, Seventh, Last;
        public WideValue(long value) => First = Second = Third = Fourth = Fifth = Sixth = Seventh = Last = value;
    }
}
