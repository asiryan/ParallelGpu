using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AdditionalValueTranslationRegressionTests(GpuFixture fixture)
{
    private static readonly decimal? StaticNullDecimal = null;
    private static readonly decimal? StaticScaledNegativeZero = new decimal(0, 0, 0, true, 17);
    private static readonly StaticMarker StaticMarkerValue = new('\uffff', true);
    private static readonly ReadonlyNativeConstant StaticNativeValue = new(17);
    private static ReadonlyNativeConstant MutableStaticValue = new(19);
    private static readonly GpuView<int> StaticNativeView = default;
    private static readonly NativeViewCarrier StaticWrappedView = new(default);

    [Fact]
    public void PublicSequentialSizeMetadataDoesNotChangeAutomaticArrayStrides()
    {
        SizedPublicValue[] input =
        [new() { Number = 3 }, new() { Number = 5 }, new() { Number = 7 }];
        SizedPublicValue[] output = new SizedPublicValue[input.Length];

        fixture.Gpu.For(0, input.Length, index =>
        {
            output[index] = input[index];
            output[index].Number += 10;
        });

        Assert.Equal(new[] { 3, 5, 7 }, input.Select(value => value.Number));
        Assert.Equal(new[] { 13, 15, 17 }, output.Select(value => value.Number));
    }

    [Fact]
    public void NestedSizedValuesAndEmptyReadonlyFieldsKeepTheirNeighbouringValues()
    {
        SizedPublicContainer[] input =
        [
            new() { Marker = 2, Value = new() { Number = 11 }, Tail = 101 },
            new() { Marker = 3, Value = new() { Number = 13 }, Tail = 103 },
            new() { Marker = 5, Value = new() { Number = 17 }, Tail = 107 }
        ];
        SizedPublicContainer[] output = new SizedPublicContainer[input.Length];
        EmptyFieldPayload[] emptyFields =
        [new() { Empty = default, Number = 19, Marker = '\uffff' }, new() { Empty = default, Number = 23, Marker = '\u8000' }, new() { Empty = default, Number = 29, Marker = '\ud800' }];
        EmptyFieldPayload[] emptyOutput = new EmptyFieldPayload[emptyFields.Length];

        fixture.Gpu.For(0, input.Length, index =>
        {
            output[index] = Copy(input[index]);
            emptyOutput[index] = Copy(emptyFields[index]);
            emptyOutput[index].Marker = unchecked((char)(emptyOutput[index].Marker + 1));
        });

        Assert.Equal(input.Select(value => value.Marker), output.Select(value => value.Marker));
        Assert.Equal(input.Select(value => value.Value.Number), output.Select(value => value.Value.Number));
        Assert.Equal(input.Select(value => value.Tail), output.Select(value => value.Tail));
        Assert.Equal(new[] { 19, 23, 29 }, emptyOutput.Select(value => value.Number));
        Assert.Equal(new[] { '\0', '\u8001', '\ud801' }, emptyOutput.Select(value => value.Marker));
    }

    [Fact]
    public void DecimalAndNullableDecimalTransportPreservesScaleSignAndAllPayloadBits()
    {
        decimal[] values =
        [new decimal(0, 0, 0, true, 17), new decimal(123, 0, 0, false, 4), decimal.MaxValue, decimal.MinValue];
        DecimalPayload[] input = values.Select((value, index) => new DecimalPayload
        {
            Amount = value,
            Optional = index == 1 ? null : value,
            Marker = (char)(0x8000 + index)
        }).ToArray();
        DecimalPayload[] output = new DecimalPayload[input.Length];
        decimal captured = values[0];
        decimal[] captures = new decimal[input.Length];
        Action<int> body = index =>
        {
            output[index] = Copy(input[index]);
            captures[index] = captured;
        };

        fixture.Gpu.For(0, input.Length, body);
        AssertDecimalPayloads(input, output);
        Assert.All(captures, value => Assert.Equal(decimal.GetBits(captured), decimal.GetBits(value)));

        captured = values[1];
        fixture.Gpu.For(0, input.Length, body);
        AssertDecimalPayloads(input, output);
        Assert.All(captures, value => Assert.Equal(decimal.GetBits(captured), decimal.GetBits(value)));
    }

    [Fact]
    public void StaticReadonlyNullableDecimalMirrorsPreserveAbsentAndScaledZeroValues()
    {
        decimal?[] output = [1m, 2m];
        Action<int> body = index => output[index] = ReadStaticDecimal(index == 0);

        fixture.Gpu.For(0, output.Length, body);
        Assert.Null(output[0]);
        Assert.True(output[1].HasValue);
        Assert.Equal(decimal.GetBits(StaticScaledNegativeZero!.Value), decimal.GetBits(output[1]!.Value));

        output[0] = 3m;
        output[1] = 4m;
        fixture.Gpu.For(0, output.Length, body);
        Assert.Null(output[0]);
        Assert.Equal(decimal.GetBits(StaticScaledNegativeZero!.Value), decimal.GetBits(output[1]!.Value));
    }

    [Fact]
    public void ReadonlyStaticMappedValuesSupportTheirReadonlyInstanceMembersInHelpers()
    {
        int expected = ReadStaticMarker();
        int[] output = [-1, -1, -1];

        fixture.Gpu.For(0, output.Length, index => output[index] = ReadStaticMarker());

        Assert.Equal(Enumerable.Repeat(expected, output.Length), output);
    }

    [Fact]
    public void ReadonlyStaticNativeValuesAlsoUseLocalCopiesForInstanceMembers()
    {
        int[] output = [-1, -1];

        fixture.Gpu.For(0, output.Length, index => output[index] = ReadStaticNative());

        Assert.Equal(new[] { 34, 34 }, output);
    }

    [Fact]
    public void ReadonlyStaticScalarAddressesCannotEscapeOrBeMutatedThroughHelpers()
    {
        int expected = ReadStaticMarker();
        int[] output = [41];

        var escape = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1,
            index => output[index] = EscapedStaticMarker().Read()));
        Assert.Contains("read-only", escape.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(41, output[0]);

        var mutation = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, index =>
        {
            output[index] = 99;
            MutateStaticMarker();
        }));
        Assert.Contains("read-only", mutation.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(expected, ReadStaticMarker());
        Assert.Equal(41, output[0]);

        fixture.Gpu.For(0, 1, index => output[index] = ReadStaticMarker());
        Assert.Equal(expected, output[0]);
    }

    [Fact]
    public void MutableStaticValuesAndImplicitNativeViewAddressesRemainRejectedInHelpers()
    {
        int[] output = [41];

        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1,
            index => output[index] = ReadMutableStatic()));
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1,
            index => output[index] = ReadStaticNativeViewLength()));
        Assert.Equal(41, output[0]);
        Assert.Equal(38, ReadMutableStatic());

        fixture.Gpu.For(0, 1, index => output[index] = ReadStaticNative());
        Assert.Equal(34, output[0]);
    }

    [Fact]
    public void ByValueHelperReturnsCannotHideReadonlyStaticNativeViewsFromOwnerValidation()
    {
        int[] output = [41];

        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, index =>
        {
            output[index] = 99;
            output[index] = HiddenStaticNativeView().Length;
        }));
        Assert.Contains("static", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(41, output[0]);

        using var buffer = fixture.Gpu.Upload(new[] { 7 });
        GpuView<int> visible = buffer.DeviceView;
        fixture.Gpu.For(0, 1, index => output[index] = visible[index] + visible.Length);
        Assert.Equal(8, output[0]);
    }

    [Fact]
    public void ValueWrappersAndTransitiveHelpersCannotHideReadonlyStaticNativeViews()
    {
        int[] output = [41];

        var direct = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, index =>
        {
            output[index] = 99;
            output[index] = HiddenWrappedNativeView().Length;
        }));
        Assert.Contains("static", direct.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(41, output[0]);

        var transitive = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, index =>
        {
            output[index] = 99;
            output[index] = TransitiveWrappedNativeView().Length;
        }));
        Assert.Contains("static", transitive.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(41, output[0]);

        fixture.Gpu.For(0, 1, index => output[index] = ReadStaticNative());
        Assert.Equal(34, output[0]);
    }

    private static T Copy<T>(T value) => value;
    private static decimal? ReadStaticDecimal(bool absent) => absent ? StaticNullDecimal : StaticScaledNegativeZero;
    private static int ReadStaticMarker() => StaticMarkerValue.Read();
    private static int ReadStaticNative() => StaticNativeValue.Read();
    private static int ReadMutableStatic() => MutableStaticValue.Read();
    private static int ReadStaticNativeViewLength() => StaticNativeView.Length;
    private static GpuView<int> HiddenStaticNativeView() => StaticNativeView;
    private static GpuView<int> HiddenWrappedNativeView() => StaticWrappedView.View;
    private static GpuView<int> TransitiveWrappedNativeView() => HiddenWrappedNativeView();
    private static ref readonly StaticMarker EscapedStaticMarker() => ref StaticMarkerValue;
    private static void MutateStaticMarker() => Unsafe.AsRef(in StaticMarkerValue) = default;

    private static void AssertDecimalPayloads(DecimalPayload[] expected, DecimalPayload[] actual)
    {
        for (int index = 0; index < expected.Length; index++)
        {
            Assert.Equal(decimal.GetBits(expected[index].Amount), decimal.GetBits(actual[index].Amount));
            Assert.Equal(expected[index].Optional.HasValue, actual[index].Optional.HasValue);
            if (expected[index].Optional.HasValue)
                Assert.Equal(decimal.GetBits(expected[index].Optional!.Value), decimal.GetBits(actual[index].Optional!.Value));
            Assert.Equal(expected[index].Marker, actual[index].Marker);
        }
    }

    private readonly struct EmptyValue;

    private readonly struct NativeViewCarrier(GpuView<int> view)
    {
        public readonly GpuView<int> View = view;
    }

    private readonly struct StaticMarker(char marker, bool enabled)
    {
        private readonly char marker = marker;
        private readonly bool enabled = enabled;

        public int Read() => marker + (enabled ? 65536 : 0);
    }

    private struct EmptyFieldPayload
    {
        public EmptyValue Empty;
        public int Number;
        public char Marker;
    }

    private struct DecimalPayload
    {
        public decimal Amount;
        public decimal? Optional;
        public char Marker;
    }
}

[StructLayout(LayoutKind.Sequential, Size = 32)]
public struct SizedPublicValue
{
    public int Number;
}

public struct SizedPublicContainer
{
    public byte Marker;
    public SizedPublicValue Value;
    public long Tail;
}

public readonly struct ReadonlyNativeConstant(int number)
{
    public readonly int Number = number;

    public int Read() => Number * 2;
}
