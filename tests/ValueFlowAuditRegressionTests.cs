using System.Runtime.InteropServices;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ValueFlowAuditRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void GenericRefReplacementPreservesMappedNestedValuesAndIndependentCopies()
    {
        FlowPayload[] input =
        [
            new() { Enabled = true, Marker = '\uffff', Code = WideCode.High },
            new() { Enabled = false, Marker = '\ud800', Code = null },
            new() { Enabled = true, Marker = '\u8000', Code = WideCode.Low }
        ];
        var initial = new FlowPayload { Enabled = false, Marker = '?', Code = WideCode.High };
        FlowPayload[] destination = Enumerable.Repeat(initial, input.Length).ToArray();
        FlowPayload[] previous = new FlowPayload[input.Length];

        fixture.Gpu.For(0, input.Length, index =>
        {
            previous[index] = Replace(ref destination[index], input[index]);
            if (destination[index].Enabled)
                destination[index].Marker = unchecked((char)(destination[index].Marker + 1));
        });

        Assert.Equal(Enumerable.Repeat(initial, input.Length), previous);
        Assert.Equal(new[] { true, false, true }, destination.Select(value => value.Enabled));
        Assert.Equal(new[] { '\0', '\ud800', '\u8001' }, destination.Select(value => value.Marker));
        Assert.Equal(input.Select(value => value.Code), destination.Select(value => value.Code));
        Assert.Equal(new[] { '\uffff', '\ud800', '\u8000' }, input.Select(value => value.Marker));
    }

    [Fact]
    public void NullableRefHelpersRollBackAllWritesAndRecoverWithTheCachedBody()
    {
        char?[] values = ['\uffff', '\ud800'];
        int[] observed = [41, 43];
        Action<int> body = index =>
        {
            observed[index] = 99;
            ref char? value = ref Element(values, index);
            char marker = ReadMarker(ref value);
            value = unchecked((char)(marker + 1));
            observed[index] = marker;
        };

        fixture.Gpu.For(0, values.Length, body);
        Assert.Equal(new char?[] { '\0', '\ud801' }, values);
        Assert.Equal(new[] { 65535, 55296 }, observed);

        values[1] = null;
        char?[] beforeValues = (char?[])values.Clone();
        int[] beforeObserved = (int[])observed.Clone();
        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, values.Length, body));
        Assert.Equal(beforeValues, values);
        Assert.Equal(beforeObserved, observed);

        values[1] = '\u8000';
        fixture.Gpu.For(0, values.Length, body);
        Assert.Equal(new char?[] { '\u0001', '\u8001' }, values);
        Assert.Equal(new[] { 0, 32768 }, observed);
    }

    [Fact]
    public void PrimitiveCharacterHelpersRetainFullUtf16ValuesThroughInstanceFieldReads()
    {
        char[] left = ['\0', '\uffff', '\u8000', '\ud800', '\udfff'];
        char[] right = ['\uffff', '\0', '\u7fff', '\udfff', '\ud800'];
        int[] comparison = new int[left.Length];
        int[] hashes = new int[left.Length];
        bool[] equality = new bool[left.Length];

        fixture.Gpu.For(0, left.Length, index =>
        {
            comparison[index] = CompareMarkers(left[index], right[index]);
            hashes[index] = MarkerHash(left[index]);
            equality[index] = EqualMarkers(left[index], right[index]);
        });

        Assert.Equal(left.Zip(right, (first, second) => first.CompareTo(second)), comparison);
        Assert.Equal(left.Select(value => value.GetHashCode()), hashes);
        Assert.Equal(left.Zip(right, (first, second) => first.Equals(second)), equality);
    }

    [Fact]
    public void GenericReadOnlyElementHelpersPreserveMappedValuesAndGuardedBounds()
    {
        FlowPayload[] input =
        [
            new() { Enabled = true, Marker = '\uffff', Code = WideCode.High },
            new() { Enabled = false, Marker = '\ud800', Code = null }
        ];
        FlowPayload[] output = new FlowPayload[input.Length];
        fixture.Gpu.For(0, input.Length, index => output[index] = ReadOnlyElement(input, index));
        Assert.Equal(input, output);

        int[] journal = [41];
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, index =>
        {
            journal[index] = 99;
            output[index] = ReadOnlyElement(input, input.Length);
        }));
        Assert.Equal(41, journal[0]);
        Assert.Equal(input, output);
    }

    [Fact]
    public unsafe void SizeOfRetainsTheClrLayoutInLoopsAndTranslatedHelpers()
    {
        var value = new SizedPayload { Prefix = 3, Enabled = true, Marker = '\uffff', Number = 17 };
        int expected = sizeof(SizedPayload);
        int divisor = 3;
        int[] sizes = new int[5];

        fixture.Gpu.For(0, 1, _ =>
        {
            sizes[0] = sizeof(SizedPayload);
            sizes[1] = PayloadSize(value);
            sizes[2] = sizeof(SizedPayload) / divisor;
            sizes[3] = NoArgumentPayloadSize();
            sizes[4] = GenericSize<SizedPayload>();
        });

        Assert.Equal(new[] { expected, expected, expected / divisor, expected, expected }, sizes);
    }

    [Fact]
    public void ByRefBooleanCharacterAndNullableEnumFieldsPreserveTheMappedStorage()
    {
        FlowPayload[] input =
        [
            new() { Enabled = true, Marker = '\uffff', Code = WideCode.High },
            new() { Enabled = false, Marker = '\ud800', Code = null }
        ];

        fixture.Gpu.For(0, input.Length, index =>
            ChangeFields(ref input[index].Enabled, ref input[index].Marker, ref input[index].Code));

        Assert.Equal(new[] { false, true }, input.Select(value => value.Enabled));
        Assert.Equal(new[] { '\0', '\ud801' }, input.Select(value => value.Marker));
        Assert.Equal(new WideCode?[] { WideCode.High | WideCode.Low, WideCode.Low }, input.Select(value => value.Code));
    }

    private static T Replace<T>(ref T target, T replacement)
    {
        T previous = target;
        target = replacement;
        return previous;
    }

    private static ref T Element<T>(T[] values, int index) => ref values[index];
    private static ref readonly T ReadOnlyElement<T>(T[] values, int index) => ref values[index];
    private static unsafe int PayloadSize(SizedPayload value) => sizeof(SizedPayload) + value.Prefix - value.Prefix;
    private static unsafe int NoArgumentPayloadSize() => sizeof(SizedPayload);
    private static unsafe int GenericSize<T>() where T : unmanaged => sizeof(T);
    private static void ChangeFields(ref bool enabled, ref char marker, ref WideCode? code)
    {
        enabled = !enabled;
        marker = unchecked((char)(marker + 1));
        code = code.HasValue ? code.Value | WideCode.Low : WideCode.Low;
    }
    private static char ReadMarker(ref char? value) => value!.Value;
    private static int CompareMarkers(char left, char right) => left.CompareTo(right);
    private static int MarkerHash(char value) => value.GetHashCode();
    private static bool EqualMarkers(char left, char right) => left.Equals(right);

    private enum WideCode : ulong { Low = 1, High = 1UL << 63 }

    private struct FlowPayload
    {
        public bool Enabled;
        public char Marker;
        public WideCode? Code;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct SizedPayload
    {
        public byte Prefix;
        public bool Enabled;
        public char Marker;
        public long Number;
    }
}
