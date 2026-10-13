using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class NextNumericAuditTests(GpuFixture fixture)
{
    [Fact]
    public void VariableInt32ShiftCountsFollowCilMaskingInDirectAndHelperOperations()
    {
        int[] counts = [-1, 0, 1, 15, 31, 32, 33, 63, 64, 65, int.MinValue, int.MaxValue];
        int[] seeds = [1, int.MinValue, int.MaxValue, unchecked((int)0x80000001)];
        int[] values = seeds.SelectMany(value => Enumerable.Repeat(value, counts.Length)).ToArray();
        int[] shifts = seeds.SelectMany(_ => counts).ToArray();
        int[] left = new int[values.Length], right = new int[values.Length], logicalRight = new int[values.Length];
        uint[] unsignedLeft = new uint[values.Length], unsignedRight = new uint[values.Length];
        uint[] helper = new uint[values.Length];
        fixture.Gpu.For(0, values.Length, i =>
        {
            left[i] = values[i] << shifts[i];
            right[i] = values[i] >> shifts[i];
            logicalRight[i] = values[i] >>> shifts[i];
            unsignedLeft[i] = unchecked((uint)values[i]) << shifts[i];
            unsignedRight[i] = unchecked((uint)values[i]) >> shifts[i];
            helper[i] = ShiftUnsigned32(unchecked((uint)values[i]), shifts[i]);
        });
        Assert.Equal(values.Select((value, i) => value << shifts[i]), left);
        Assert.Equal(values.Select((value, i) => value >> shifts[i]), right);
        Assert.Equal(values.Select((value, i) => value >>> shifts[i]), logicalRight);
        Assert.Equal(values.Select((value, i) => unchecked((uint)value) << shifts[i]), unsignedLeft);
        Assert.Equal(values.Select((value, i) => unchecked((uint)value) >> shifts[i]), unsignedRight);
        Assert.Equal(values.Select((value, i) => ShiftUnsigned32(unchecked((uint)value), shifts[i])), helper);
    }

    [Fact]
    public void VariableInt64ShiftCountsFollowCilMaskingInDirectAndHelperOperations()
    {
        int[] counts = [-1, 0, 1, 31, 32, 33, 63, 64, 65, 127, 128, int.MinValue, int.MaxValue];
        long[] seeds = [1, long.MinValue, long.MaxValue, unchecked((long)0x8000000000000001UL)];
        long[] values = seeds.SelectMany(value => Enumerable.Repeat(value, counts.Length)).ToArray();
        int[] shifts = seeds.SelectMany(_ => counts).ToArray();
        long[] left = new long[values.Length], right = new long[values.Length], logicalRight = new long[values.Length];
        ulong[] unsignedLeft = new ulong[values.Length], unsignedRight = new ulong[values.Length];
        ulong[] helper = new ulong[values.Length];
        fixture.Gpu.For(0, values.Length, i =>
        {
            left[i] = values[i] << shifts[i];
            right[i] = values[i] >> shifts[i];
            logicalRight[i] = values[i] >>> shifts[i];
            unsignedLeft[i] = unchecked((ulong)values[i]) << shifts[i];
            unsignedRight[i] = unchecked((ulong)values[i]) >> shifts[i];
            helper[i] = ShiftUnsigned64(unchecked((ulong)values[i]), shifts[i]);
        });
        Assert.Equal(values.Select((value, i) => value << shifts[i]), left);
        Assert.Equal(values.Select((value, i) => value >> shifts[i]), right);
        Assert.Equal(values.Select((value, i) => value >>> shifts[i]), logicalRight);
        Assert.Equal(values.Select((value, i) => unchecked((ulong)value) << shifts[i]), unsignedLeft);
        Assert.Equal(values.Select((value, i) => unchecked((ulong)value) >> shifts[i]), unsignedRight);
        Assert.Equal(values.Select((value, i) => ShiftUnsigned64(unchecked((ulong)value), shifts[i])), helper);
    }

    [Fact]
    public void NarrowAndMappedCharacterShiftHelpersUseTheInt32EvaluationStack()
    {
        sbyte[] signed = [sbyte.MinValue, -1, sbyte.MaxValue, 1];
        char[] characters = ['\uffff', '\u8000', '\ud800', 'A'];
        int[] counts = [-1, 32, 33, 63];
        int[] direct = new int[signed.Length], helper = new int[signed.Length];
        char[] mapped = new char[characters.Length];
        fixture.Gpu.For(0, signed.Length, i =>
        {
            int selected = i % 2 == 0 ? signed[i] : characters[i];
            direct[i] = selected << counts[i];
            helper[i] = ShiftNarrow(signed[i], characters[i], i % 2 == 0, counts[i]);
            mapped[i] = unchecked((char)(characters[i] << counts[i]));
        });
        Assert.Equal(signed.Select((value, i) => (i % 2 == 0 ? (int)value : characters[i]) << counts[i]), direct);
        Assert.Equal(signed.Select((value, i) => ShiftNarrow(value, characters[i], i % 2 == 0, counts[i])), helper);
        Assert.Equal(characters.Select((value, i) => unchecked((char)(value << counts[i]))), mapped);
    }

    private static uint ShiftUnsigned32(uint value, int count) => unchecked((value << count) + (value >> count));
    private static ulong ShiftUnsigned64(ulong value, int count) => unchecked((value << count) + (value >> count));
    private static int ShiftNarrow(sbyte value, char character, bool signed, int count)
    {
        int selected = signed ? value : character;
        return selected << count;
    }
}
