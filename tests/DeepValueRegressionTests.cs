using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class DeepValueRegressionTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void PublicAndPrivateAutoLayoutsRoundTripTheirFields()
    {
        AuditPublicAuto[] input = [new() { Marker = 1, Number = 11 }, new() { Marker = 2, Number = 22 }];
        AuditPublicAuto[] output = new AuditPublicAuto[input.Length];
        Gpu.For(0, input.Length, i => output[i] = input[i]);
        Assert.Equal(input.Select(value => value.Marker), output.Select(value => value.Marker));
        Assert.Equal(input.Select(value => value.Number), output.Select(value => value.Number));

        PrivateAuto[] privateInput = [new() { Marker = 3, Number = 33 }, new() { Marker = 4, Number = 44 }];
        PrivateAuto[] privateOutput = new PrivateAuto[privateInput.Length];
        Gpu.For(0, privateInput.Length, i => privateOutput[i] = privateInput[i]);
        Assert.Equal(privateInput.Select(value => value.Marker), privateOutput.Select(value => value.Marker));
        Assert.Equal(privateInput.Select(value => value.Number), privateOutput.Select(value => value.Number));
    }

    [Fact]
    public void DateTimeArraysNestedFieldsAndFreshCapturesPreserveTicksAndKind()
    {
        DateTime[] input = [new(12345), new(9999999, DateTimeKind.Utc), new(456789, DateTimeKind.Local)];
        DateTime[] output = new DateTime[input.Length];
        Gpu.For(0, input.Length, i => output[i] = input[i]);
        Assert.Equal(input.Select(value => value.Ticks), output.Select(value => value.Ticks));
        Assert.Equal(input.Select(value => value.Kind), output.Select(value => value.Kind));

        DateTime captured = input[1];
        Action<int> body = i => output[i] = captured;
        Gpu.For(0, output.Length, body);
        Assert.All(output, value => { Assert.Equal(captured.Ticks, value.Ticks); Assert.Equal(captured.Kind, value.Kind); });
        captured = input[2];
        Gpu.For(0, output.Length, body);
        Assert.All(output, value => { Assert.Equal(captured.Ticks, value.Ticks); Assert.Equal(captured.Kind, value.Kind); });

        AuditAutoContainer[] nested = [new() { Date = input[2], Payload = new() { Marker = 9, Number = 90 } }];
        AuditAutoContainer[] nestedOutput = new AuditAutoContainer[1];
        Gpu.For(0, 1, i => nestedOutput[i] = nested[i]);
        Assert.Equal(nested[0].Date.Ticks, nestedOutput[0].Date.Ticks);
        Assert.Equal(nested[0].Date.Kind, nestedOutput[0].Date.Kind);
        Assert.Equal(nested[0].Payload.Marker, nestedOutput[0].Payload.Marker);
        Assert.Equal(nested[0].Payload.Number, nestedOutput[0].Payload.Number);
    }

    [Fact]
    public void PublicAndPrivateInlineArraysRejectBeforeEffects()
    {
        AuditPublicInlineArray publicValue = default;
        PrivateInlineArray privateValue = default;
        for (int i = 0; i < 4; i++)
        {
            publicValue[i] = 10 + i;
            privateValue[i] = 20 + i;
        }
        AssertRejected(publicValue);
        AssertRejected(privateValue);
        AssertRejected(new AuditInlineContainer { Payload = publicValue });
        AssertRejected(new PrivateInlineContainer { Payload = privateValue });
    }

    [Fact]
    public unsafe void PublicAndPrivateFixedBuffersRejectBeforeEffects()
    {
        AuditPublicFixedBuffer publicValue = default;
        PrivateFixedBuffer privateValue = default;
        for (int i = 0; i < 4; i++)
        {
            publicValue.Values[i] = 10 + i;
            privateValue.Values[i] = 20 + i;
        }
        AssertRejected(publicValue);
        AssertRejected(privateValue);
        AssertRejected(new AuditFixedContainer { Payload = publicValue });
        AssertRejected(new PrivateFixedContainer { Payload = privateValue });
    }

    [Fact]
    public void NativeGraphsRejectHiddenStorageBeforeAdapterCallbacks()
    {
        AssertGraphRejected<AuditPublicInlineArray>();
        AssertGraphRejected<AuditPublicFixedBuffer>();
        AssertGraphRejected<AuditInlineContainer>();
        AssertGraphRejected<AuditFixedContainer>();
    }

    [Fact]
    public void OrdinaryExplicitStructureSizeStillPreservesFields()
    {
        AuditPaddedValue[] input = [new() { Number = 11 }, new() { Number = 22 }];
        AuditPaddedValue[] output = new AuditPaddedValue[2];
        Gpu.For(0, 2, i => output[i] = input[i]);
        Assert.Equal(new[] { 11, 22 }, output.Select(value => value.Number));
    }

    private void AssertRejected<T>(T value) where T : unmanaged
    {
        T[] input = [value], output = [value];
        int[] markers = [7];
        var expectedInput = MemoryMarshal.AsBytes(input.AsSpan()).ToArray();
        var expectedOutput = MemoryMarshal.AsBytes(output.AsSpan()).ToArray();
        Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i =>
        {
            markers[i] = 99;
            output[i] = input[i];
        }));
        Assert.Throws<NotSupportedException>(() => Gpu.For<T>(0, 1, i => output[i] = input[i]));
        Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i =>
        {
            markers[i] = 99;
            IgnoreValue(value);
        }));
        Assert.Equal(new[] { 7 }, markers);
        Assert.Equal(expectedInput, MemoryMarshal.AsBytes(input.AsSpan()).ToArray());
        Assert.Equal(expectedOutput, MemoryMarshal.AsBytes(output.AsSpan()).ToArray());
        Assert.Throws<NotSupportedException>(() => GpuCollectionAdapters.Register<UnsupportedCollection<T>, T>(
            owner => new(owner.Values)));
        Assert.Throws<NotSupportedException>(() => Gpu.CreateAppendBuffer<T>(1));
        Assert.Throws<NotSupportedException>(() => Gpu.Upload(input));
    }

    private void AssertGraphRejected<T>() where T : unmanaged
    {
        var adapter = new UnsupportedGraphAdapter<T>();
        Assert.Throws<NotSupportedException>(() => Gpu.UploadObjectGraph<GraphNode, T>([new()], adapter));
        Assert.Equal(0, adapter.ReadCount);
    }

    private static void IgnoreValue<T>(T value) where T : unmanaged { }

    private sealed class UnsupportedCollection<T> where T : unmanaged
    {
        public T[] Values { get; } = new T[1];
        public T this[int index] => Values[index];
    }

    private sealed class GraphNode { }

    private sealed class UnsupportedGraphAdapter<T> : IGpuObjectGraphAdapter<GraphNode, T> where T : unmanaged
    {
        public int ReadCount { get; private set; }
        public T ReadValue(GraphNode source) { ReadCount++; return default; }
        public IReadOnlyList<GraphNode?> ReadReferences(GraphNode source) { ReadCount++; return []; }
        public GraphNode CreateObject(T value) => new();
        public void WriteValue(GraphNode target, T value) { }
        public void WriteReferences(GraphNode target, IReadOnlyList<GraphNode?> references) { }
    }

    [StructLayout(LayoutKind.Auto)]
    private struct PrivateAuto { public byte Marker; public int Number; }
    [InlineArray(4)]
    private struct PrivateInlineArray { private int element; }
    private unsafe struct PrivateFixedBuffer { public fixed int Values[4]; }
    private struct PrivateInlineContainer { public PrivateInlineArray Payload; }
    private struct PrivateFixedContainer { public PrivateFixedBuffer Payload; }
}

[StructLayout(LayoutKind.Auto)]
public struct AuditPublicAuto { public byte Marker; public int Number; }
public struct AuditAutoContainer { public DateTime Date; public AuditPublicAuto Payload; }
[InlineArray(4)]
public struct AuditPublicInlineArray { private int element; }
public unsafe struct AuditPublicFixedBuffer { public fixed int Values[4]; }
public struct AuditInlineContainer { public AuditPublicInlineArray Payload; }
public struct AuditFixedContainer { public AuditPublicFixedBuffer Payload; }
[StructLayout(LayoutKind.Sequential, Size = 16)]
public struct AuditPaddedValue { public int Number; }
