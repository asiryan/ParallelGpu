using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class NullableLoopTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void NullableElementsSupportLiftedArithmeticAndFreshNullableCaptures()
    {
        int?[] input = [null, 5, -2, 0];
        int?[] output = new int?[input.Length];
        int? offset = 3;
        Action<int> body = i => output[i] = input[i] + offset;

        Gpu.For<int?>(0, input.Length, body);
        Assert.Equal(new int?[] { null, 8, 1, 3 }, output);
        offset = null;
        Gpu.For<int?>(0, input.Length, body);
        Assert.Equal(new int?[] { null, null, null, null }, output);
        offset = -1;
        Gpu.For<int?>(0, input.Length, body);
        Assert.Equal(new int?[] { null, 4, -3, -1 }, output);
        Assert.Equal(new int?[] { null, 5, -2, 0 }, input);
    }

    [Fact]
    public void NullableLocalsConstructorsCoalescingAndDefaultArgumentsPreservePresence()
    {
        int?[] input = [null, 0, 7];
        int? captured = null;
        bool[] present = new bool[input.Length];
        int[] output = new int[input.Length];
        int?[] constructed = new int?[input.Length];
        Gpu.For(0, input.Length, i =>
        {
            int? value = input[i];
            present[i] = value.HasValue;
            output[i] = value.GetValueOrDefault(11) + (captured ?? 2);
            constructed[i] = new int?(value.GetValueOrDefault());
        });
        Assert.Equal(new[] { false, true, true }, present);
        Assert.Equal(new[] { 13, 2, 9 }, output);
        Assert.Equal(new int?[] { 0, 0, 7 }, constructed);
    }

    [Fact]
    public void NullableBooleanAndFullWidthCharacterElementsAndCapturesRoundTrip()
    {
        bool?[] flags = [null, false, true];
        bool?[] result = new bool?[flags.Length];
        bool? capturedFlag = true;
        Gpu.For<bool?>(0, flags.Length, i => result[i] = flags[i] & capturedFlag);
        Assert.Equal(new bool?[] { null, false, true }, result);

        char?[] characters = [null, '\uFFFF', '\u8000', '\uD800'];
        char?[] shifted = new char?[characters.Length];
        char? capturedCharacter = '\uFFFF';
        Gpu.For<char?>(0, characters.Length, i =>
        {
            char? value = characters[i];
            shifted[i] = value.HasValue
                ? new char?(unchecked((char)(value.GetValueOrDefault() + capturedCharacter.GetValueOrDefault())))
                : null;
        });
        Assert.Equal(new char?[] { null, '\uFFFE', '\u7FFF', '\uD7FF' }, shifted);
    }

    [Fact]
    public void NestedNullableStructureFieldsAndNullablePrivateStructuresRoundTrip()
    {
        Entry[] values = [new(1, true, '\uFFFF'), new(null, null, null), new(0, false, '\u8000')];
        Gpu.For<Entry>(0, values.Length, i =>
        {
            values[i].Count += 2;
            values[i].Enabled = !values[i].Enabled;
        });
        Assert.Equal(new Entry[] { new(3, false, '\uFFFF'), new(null, null, null), new(2, true, '\u8000') }, values);

        Pair?[] pairs = [new Pair(1, 2), null, new Pair(-3, 4)];
        Pair?[] output = new Pair?[pairs.Length];
        Gpu.For<Pair?>(0, pairs.Length, i => output[i] = Shift(pairs[i], 10));
        Assert.Equal(new Pair?[] { new Pair(11, 12), null, new Pair(7, 14) }, output);
    }

    [Fact]
    public void NullableListsAndJaggedAliasesUseSharedElementStorage()
    {
        int?[] shared = [1, null, 3];
        int?[][] rows = [shared, shared];
        IList<int?> output = new List<int?> { null, null, null };
        Gpu.For<int?>(0, shared.Length, i =>
        {
            rows[0][i] += 2;
            output[i] = rows[1][i];
        });
        Assert.Equal(new int?[] { 3, null, 5 }, shared);
        Assert.Equal(shared, output);
        Assert.Same(rows[0], rows[1]);
    }

    [Fact]
    public void GuardedValueAndLiftedComparisonsHaveClrSemantics()
    {
        int?[] left = [null, 0, 3, null];
        int?[] right = [null, 1, null, 2];
        bool[] equal = new bool[left.Length];
        bool[] less = new bool[left.Length];
        int[] guarded = new int[left.Length];
        Gpu.For(0, left.Length, i =>
        {
            equal[i] = left[i] == right[i];
            less[i] = left[i] < right[i];
            int? value = left[i];
            guarded[i] = value.HasValue ? value.Value : -1;
        });
        Assert.Equal(new[] { true, false, false, false }, equal);
        Assert.Equal(new[] { false, true, false, false }, less);
        Assert.Equal(new[] { -1, 0, 3, -1 }, guarded);
    }

    [Fact]
    public void NullValueThrowsBeforeAnyHostCollectionIsChanged()
    {
        int?[] input = [4, null];
        int[] output = [-1, -1];
        var error = Assert.Throws<InvalidOperationException>(() =>
            Gpu.For(0, input.Length, i => output[i] = ReadValue(input[i])));
        Assert.Contains("Nullable", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(new[] { -1, -1 }, output);
        Assert.Equal(new int?[] { 4, null }, input);
        Gpu.For(0, input.Length, i => output[i] = input[i].GetValueOrDefault(9));
        Assert.Equal(new[] { 4, 9 }, output);
    }

    [Fact]
    public void TypedFacadeAcceptsNullableAndInvalidHostTypesStillFailValidation()
    {
        int?[] values = [1, null];
        ParallelGpu.GpuParallel.For<int?>(0, values.Length, new() { Backend = Gpu.Backend },
            i => values[i] += 2);
        Assert.Equal(new int?[] { 3, null }, values);
        Assert.Throws<NotSupportedException>(() => Gpu.For<string>(0, 1, _ => { }));
        Assert.Throws<NotSupportedException>(() => Gpu.For<WithReference?>(0, 1, _ => { }));
    }

    [Fact]
    public void LargeNullableArraysUseConvertedSliceTransfers()
    {
        int?[] input = Enumerable.Range(0, 131_072).Select(i => i % 3 == 0 ? (int?)null : i).ToArray();
        int?[] output = new int?[input.Length];
        Gpu.For<int?>(0, input.Length, i => output[i] = input[i]);
        Assert.Equal(input, output);
    }
    [Fact]
    public void NullableNumericAndEnumPayloadsPreserveValuesAndNulls()
    {
        RoundTrip(new byte?[] { null, byte.MaxValue, 0 });
        RoundTrip(new sbyte?[] { null, sbyte.MinValue, sbyte.MaxValue });
        RoundTrip(new short?[] { null, short.MinValue, short.MaxValue });
        RoundTrip(new ushort?[] { null, ushort.MaxValue, 0 });
        RoundTrip(new uint?[] { null, uint.MaxValue, 0 });
        RoundTrip(new long?[] { null, long.MinValue, long.MaxValue });
        RoundTrip(new ulong?[] { null, ulong.MaxValue, 0 });
        RoundTrip(new float?[] { null, float.NaN, float.PositiveInfinity, -1.5f });
        RoundTrip(new double?[] { null, double.NaN, double.NegativeInfinity, -2.5 });
        RoundTrip(new Code?[] { null, Code.High, Code.Low });
    }

    private void RoundTrip<T>(T[] values)
    {
        T[] output = new T[values.Length];
        Gpu.For<T>(0, values.Length, i => output[i] = values[i]);
        Assert.Equal(values, output);
    }
    private static Pair? Shift(Pair? value, int offset)
    {
        if (!value.HasValue)
            return null;
        Pair actual = value.GetValueOrDefault();
        return new Pair(actual.X + offset, actual.Y + offset);
    }

    private static int ReadValue(int? value) => value!.Value;

    private enum Code : ushort { Low = 0, High = ushort.MaxValue }
    private record struct Pair(int X, int Y);
    private record struct Entry(int? Count, bool? Enabled, char? Code);
    private record struct WithReference(string? Text);
}
