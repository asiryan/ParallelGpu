using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class PrimitiveLoopTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void BooleanCollectionsPreserveAliasesReferencesAndFreshCaptures()
    {
        bool[] values = [true, false, true, false];
        bool[][] rows = [values, values, []];
        IReadOnlyList<bool> input = values;
        IList<bool> output = new List<bool> { false, false, false, false };
        bool expected = true;
        Action<int> body = i =>
        {
            Toggle(ref rows[0][i]);
            output[i] = input[i] == expected && rows[2].Length == 0;
        };

        Gpu.For<bool>(0, values.Length, body);
        Assert.Equal(new[] { false, true, false, true }, values);
        Assert.Equal(new[] { false, true, false, true }, output);
        Assert.Same(values, rows[1]);

        expected = false;
        Gpu.For<bool>(0, values.Length, body);
        Assert.Equal(new[] { true, false, true, false }, values);
        Assert.Equal(new[] { false, true, false, true }, output);
    }

    [Fact]
    public void CharacterCollectionsPreserveUtf16ValuesAndFreshCaptures()
    {
        char[,] values = { { '\0', '\uFFFF', '\uD800' }, { 'A', '\u8000', '\uDFFF' } };
        var output = new List<char> { '\0', '\0', '\0', '\0', '\0', '\0' };
        char offset = '\uFFFF';
        Action<int> body = row =>
        {
            for (int column = 0; column < values.GetLength(1); column++)
            {
                Increment(ref values[row, column]);
                char value = values[row, column];
                output[row * values.GetLength(1) + column] = ShiftThroughHelper(value, offset);
            }
        };

        Gpu.For<char>(0, values.GetLength(0), body);
        Assert.Equal(new[] { '\u0001', '\0', '\uD801', 'B', '\u8001', '\uE000' }, values.Cast<char>());
        Assert.Equal(new[] { '\0', '\uFFFF', '\uD800', 'A', '\u8000', '\uDFFF' }, output);

        offset = '\u0001';
        Gpu.For<char>(0, values.GetLength(0), body);
        Assert.Equal(new[] { '\u0003', '\u0002', '\uD803', 'D', '\u8003', '\uE002' }, output);
    }

    [Fact]
    public void EnumsSupportAllUnderlyingTypesAndFlagOperations()
    {
        RoundTrip<SignedByte>(unchecked((SignedByte)sbyte.MinValue));
        RoundTrip<UnsignedByte>((UnsignedByte)byte.MaxValue);
        RoundTrip<SignedShort>(unchecked((SignedShort)short.MinValue));
        RoundTrip<UnsignedShort>((UnsignedShort)ushort.MaxValue);
        RoundTrip<SignedInt>(unchecked((SignedInt)int.MinValue));
        RoundTrip<UnsignedInt>((UnsignedInt)uint.MaxValue);
        RoundTrip<SignedLong>(unchecked((SignedLong)long.MinValue));
        RoundTrip<UnsignedLong>((UnsignedLong)ulong.MaxValue);
        RoundTrip<bool>(true);
        RoundTrip<char>('\uFFFF');

        UnsignedLong[] values = [UnsignedLong.High, 0, UnsignedLong.High | UnsignedLong.Low];
        UnsignedLong[][] rows = [values];
        IList<UnsignedLong> output = new List<UnsignedLong> { 0, 0, 0 };
        UnsignedLong mask = UnsignedLong.Low;
        Gpu.For<UnsignedLong>(0, values.Length, i =>
        {
            AddLowFlag(ref rows[0][i]);
            output[i] = values[i] & ~mask;
        });
        Assert.Equal(new[] { UnsignedLong.High | UnsignedLong.Low, UnsignedLong.Low, UnsignedLong.High | UnsignedLong.Low }, values);
        Assert.Equal(new UnsignedLong[] { UnsignedLong.High, 0, UnsignedLong.High }, output);

        bool[] present = new bool[values.Length];
        Gpu.For(0, present.Length, i => present[i] = (values[i] & mask) != 0);
        Assert.All(present, Assert.True);
    }

    private void RoundTrip<T>(T value) where T : unmanaged
    {
        T[] input = [default, value, value, default];
        T[] output = new T[input.Length];
        Gpu.For<T>(0, output.Length, i => output[i] = input[i]);
        Assert.Equal(input, output);
    }

    private static void Toggle(ref bool value) => value = !value;

    private static void Increment(ref char value) => value = unchecked((char)(value + 1));

    private static char ShiftThroughHelper(char value, char offset) => Shift(value, offset);

    private static char Shift(char value, char offset) => unchecked((char)(value + offset));

    private static void AddLowFlag(ref UnsignedLong value) => value |= UnsignedLong.Low;

    private enum SignedByte : sbyte { }
    private enum UnsignedByte : byte { }
    private enum SignedShort : short { }
    private enum UnsignedShort : ushort { }
    private enum SignedInt : int { }
    private enum UnsignedInt : uint { }
    private enum SignedLong : long { }
    [Flags]
    private enum UnsignedLong : ulong { Low = 1, High = 1UL << 63 }
}
