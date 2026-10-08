using System.Numerics;
using System.Runtime.InteropServices;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class StructLoopTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void ComplexLocalsConstructorsOperatorsAndFreshCaptures()
    {
        Complex[] input = [new(1, 2), new(-3, 4)];
        Complex[] output = new Complex[2];
        Complex factor = new(2, -1);
        Action<int> body = i =>
        {
            Complex value = input[i];
            output[i] = value * factor + new Complex(i, factor.Real);
        };

        Gpu.For<Complex>(0, output.Length, body);
        Assert.Equal(new Complex[] { new(4, 5), new(-1, 13) }, output);
        factor = new(1, 3);
        Gpu.For<Complex>(0, output.Length, body);
        Assert.Equal(new Complex[] { new(-5, 6), new(-14, -4) }, output);
        Assert.Equal(new Complex[] { new(1, 2), new(-3, 4) }, input);
    }

    [Fact]
    public void PrivateNestedStructuresSupportFieldsPropertiesAndRefMutation()
    {
        Item[] shared = [new(new(1, 2), 3), new(new(4, 5), 6)];
        Item[][] rows = [shared, shared, []];
        IReadOnlyList<Item> input = shared;
        IList<Item> output = new List<Item> { default, default };
        Pair offset = new(10, 20);
        Action<int> body = i =>
        {
            rows[0][i].Position.X += offset.X;
            Move(ref rows[1][i].Position, offset);
            Item value = input[i];
            value.Weight += value.Position.Sum;
            output[i] = new Item(value.Position + offset, value.Weight);
        };

        Gpu.For<Item>(0, shared.Length, body);
        Assert.Equal(new Item[] { new(new(21, 22), 3), new(new(24, 25), 6) }, shared);
        Assert.Equal(new Item[] { new(new(31, 42), 46), new(new(34, 45), 55) }, output);
        Assert.Same(shared, rows[1]);

        offset = new(1, 2);
        Gpu.For<Item>(0, shared.Length, body);
        Assert.Equal(new Item[] { new(new(23, 24), 3), new(new(26, 27), 6) }, shared);
    }

    [Fact]
    public void StructRectangularArraysAndInstanceMethodsUpdateOriginalElements()
    {
        Pair[,] values = { { new(1, 2), new(3, 4) }, { new(5, 6), new(7, 8) } };
        Pair[] output = new Pair[4];
        Gpu.For<Pair>(0, values.GetLength(0), row =>
        {
            for (int column = 0; column < values.GetLength(1); column++)
            {
                values[row, column].Translate(new Pair(10, 20));
                output[row * values.GetLength(1) + column] = values[row, column];
            }
        });
        Assert.Equal(new Pair[] { new(11, 22), new(13, 24), new(15, 26), new(17, 28) }, output);
        Assert.Equal(output, values.Cast<Pair>());
    }

    [Fact]
    public void StructBooleanAndCharacterFieldsPreserveFullValues()
    {
        Tagged[] values = [new(1, true, '\uFFFF'), new(2, false, '\u8000'), new(3, true, '\uD800')];
        Tagged[] output = new Tagged[3];
        char offset = '\u0001';
        Gpu.For<Tagged>(0, values.Length, i =>
        {
            Tagged value = values[i];
            value.Enabled = !value.Enabled;
            value.Code = unchecked((char)(value.Code + offset));
            value.Index += 10;
            output[i] = value;
        });
        Assert.Equal(new Tagged[] { new(11, false, '\0'), new(12, true, '\u8001'), new(13, false, '\uD801') }, output);
        Assert.Equal(new Tagged[] { new(1, true, '\uFFFF'), new(2, false, '\u8000'), new(3, true, '\uD800') }, values);
    }

    [Fact]
    public void MixedStructAndPrimitiveCollectionsAndFacadeUseTheSameBody()
    {
        Pair[] input = [new(1, 2), new(3, 4)];
        int[] output = new int[2];
        Gpu.For(0, output.Length, i => output[i] = input[i].Sum);
        Assert.Equal(new[] { 3, 7 }, output);

        Pair[] result = new Pair[2];
        ParallelGpu.GpuParallel.For<Pair>(0, result.Length, new() { Backend = Gpu.Backend },
            i => result[i] = input[i] + new Pair(10, 20));
        Assert.Equal(new Pair[] { new(11, 22), new(13, 24) }, result);
    }

    [Fact]
    public void ManagedFieldsAndUnsupportedMethodsAreRejectedBeforeHostWrites()
    {
        WithReference[] input = [new() { Value = "text" }];
        int[] output = [-1];
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For(0, output.Length, i => output[i] = input[i].Value.Length));
        Assert.Equal(new[] { -1 }, output);

        BigInteger[] big = [BigInteger.One];
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For(0, output.Length, i => output[i] = (int)big[i]));
        Assert.Equal(new[] { -1 }, output);

        Pair[] pairs = [new(1, 2)];
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For(0, output.Length, i => output[i] = pairs[i].ManagedOperation()));
        Assert.Equal(new[] { -1 }, output);
        Gpu.For(0, output.Length, i => output[i] = pairs[i].Sum);
        Assert.Equal(new[] { 3 }, output);
    }

    [Fact]
    public void ReadonlyGenericStructuresConvertPrivateBooleanAndCharacterFields()
    {
        KeyValuePair<char, bool>[] input = [new('\uFFFF', true), new('\u8000', false)];
        KeyValuePair<char, bool>[] output = new KeyValuePair<char, bool>[2];
        KeyValuePair<char, bool> offset = new('\u0001', true);
        Gpu.For<KeyValuePair<char, bool>>(0, output.Length, i =>
        {
            var value = input[i];
            output[i] = new KeyValuePair<char, bool>(unchecked((char)(value.Key + offset.Key)),
                value.Value != offset.Value);
        });
        Assert.Equal(new KeyValuePair<char, bool>[] { new('\0', false), new('\u8001', true) }, output);
        Assert.Equal(new KeyValuePair<char, bool>[] { new('\uFFFF', true), new('\u8000', false) }, input);
    }

    [Fact]
    public void MappedStructuresSupportConstructorsAndTransitiveCharacterHelpers()
    {
        Tagged[] output = new Tagged[3];
        Tagged initial = new(10, true, '\uFFFF');
        Gpu.For<Tagged>(0, output.Length, i =>
        {
            Tagged value = new(initial.Index + i, initial.Enabled, initial.Code);
            value.Advance();
            output[i] = value;
        });
        Assert.Equal(new Tagged[] { new(11, false, '\0'), new(12, false, '\0'), new(13, false, '\0') }, output);
        Assert.Equal(new Tagged(10, true, '\uFFFF'), initial);
    }

    [Fact]
    public void CapturedValuesCannotBeMutatedThroughFieldsMethodsOrReferences()
    {
        int[] output = [-1];
        Pair captured = new(1, 2);
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For(0, output.Length, i => { output[i] = 3; captured.X++; }));
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For(0, output.Length, i => { output[i] = 3; captured.Translate(new Pair(1, 2)); }));
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For(0, output.Length, i => { output[i] = 3; Move(ref captured, new Pair(1, 2)); }));
        int scalar = 1;
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For(0, output.Length, i => { output[i] = 3; Increment(ref scalar); }));
        Assert.Equal(new[] { -1 }, output);
        Assert.Equal(new Pair(1, 2), captured);
        Assert.Equal(1, scalar);
    }

    [Fact]
    public void StructureReferencesAndReadonlyHelperValuesUseMappedRepresentations()
    {
        Tagged[] values = [default, default];
        Gpu.For<Tagged>(0, values.Length, i =>
        {
            ref Tagged value = ref Identity(ref values[i]);
            value = Tagged.CreateDefault();
            Advance(ref value.Code);
        });
        Assert.Equal(new Tagged[] { new(7, true, '\0'), new(7, true, '\0') }, values);
    }

    [Fact]
    public void PackedHostLayoutsAndNativePublicStructuresPreserveFields()
    {
        Packed[] input = [new() { Marker = 7, Value = 123.5 }, new() { Marker = 9, Value = -2.25 }];
        Packed[] output = new Packed[2];
        Gpu.For<Packed>(0, output.Length, i => output[i] = input[i]);
        Assert.Equal(input, output);

        (int Index, double Value)[] native = [(1, 2.5), (3, 4.5)];
        Gpu.For<(int, double)>(0, native.Length, i =>
        {
            native[i].Index += 10;
            native[i].Value *= 2;
        });
        Assert.Equal(new[] { (11, 5.0), (13, 9.0) }, native);
    }

    [Fact]
    public void ExplicitLayoutsAreRejectedBeforeHostWrites()
    {
        Overlapping[] input = [default];
        int[] output = [-1];
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For<Overlapping>(0, input.Length, i => input[i].Integer = i));
        Assert.Throws<NotSupportedException>(() =>
            Gpu.For(0, output.Length, i => output[i] = input[i].Integer));
        Assert.Equal(new[] { -1 }, output);
    }

    [Fact]
    public void ReadonlyCapturedMembersSupportConditionalArgumentsAndSurviveCollection()
    {
        Pair captured = new(1, 2);
        int[] output = new int[2];
        Action<int> body = i => output[i] = captured.SumPlus(i == 0 ? 10 : 20);
        Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 13, 23 }, output);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        captured = new(3, 4);
        Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 17, 27 }, output);
    }

    [Fact]
    public void MappedLoopsSurviveExecutorDisposalAndFinalization()
    {
        for (int pass = 0; pass < 3; pass++)
        {
            RunOnNewExecutor(Gpu.Backend);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void RunOnNewExecutor(GpuBackend backend)
    {
        using var executor = new Gpu(new() { Backend = backend });
        Tagged[] output = new Tagged[2];
        Tagged initial = new(10, true, '\uFFFF');
        executor.For<Tagged>(0, output.Length, i =>
        {
            output[i] = new Tagged(initial.Index + i, initial.Enabled, initial.Code);
            output[i].Advance();
        });
        Assert.Equal(new Tagged[] { new(11, false, '\0'), new(12, false, '\0') }, output);
    }

    private static ref T Identity<T>(ref T value) where T : unmanaged => ref value;
    private static void Advance(ref char value) => value = unchecked((char)(value + 1));

    private static void Increment(ref int value) => value++;

    private static void Move(ref Pair value, Pair offset) => value.Translate(offset);

    private struct Pair(int x, int y)
    {
        public int X = x;
        public int Y = y;
        public readonly int Sum => X + Y;
        public readonly int SumPlus(int offset) => Sum + offset;
        public void Translate(Pair offset) { X += offset.X; Y += offset.Y; }
        public int ManagedOperation() => X.ToString().Length;
        public static Pair operator +(Pair left, Pair right) => new(left.X + right.X, left.Y + right.Y);
    }

    private struct Item(Pair position, int weight)
    {
        public Pair Position = position;
        public int Weight = weight;
    }

    private struct Tagged(int index, bool enabled, char code)
    {
        private static readonly Tagged initial = new(7, true, '\uFFFF');
        public static Tagged CreateDefault() => initial;
        public int Index = index;
        public bool Enabled = enabled;
        public char Code = code;
        public void Advance()
        {
            Index++;
            Enabled = !Enabled;
            Code = Shift(Code);
        }
        private static char Shift(char value) => unchecked((char)(value + 1));
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct Packed
    {
        public byte Marker;
        public double Value;
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct Overlapping
    {
        [FieldOffset(0)] public int Integer;
        [FieldOffset(0)] public float Real;
    }

    private struct WithReference
    {
        public string Value;
    }
}
