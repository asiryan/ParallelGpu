using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class NextTranslationAuditTests(GpuFixture fixture)
{
    private static volatile int staticBias;
    private static volatile bool staticEnabled;
    private static volatile char staticMarker;
    private static int[] staticInput = [], staticOutput = [];

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void VolatileScalarCapturePathsReadSnapshotsAndRefreshCachedBodies(bool methodGroup)
    {
        var state = new CaptureState { Input = [2, 3], Output = [41, 43], Bias = 5 };
        var wrapper = new CaptureWrapper { State = state };
        Action<int> body = methodGroup ? state.Execute :
            i => wrapper.State.Output[i] = wrapper.State.Input[i] + wrapper.State.Bias;
        fixture.Gpu.For(0, state.Output.Length, body);
        Assert.Equal(new[] { 7, 8 }, state.Output);
        int[] previousOutput = state.Output;
        state.Input = [11, 13];
        state.Output = [47, 53];
        state.Bias = 9;
        fixture.Gpu.For<int>(0, state.Output.Length, body);
        Assert.Equal(new[] { 20, 22 }, state.Output);
        Assert.Equal(new[] { 7, 8 }, previousOutput);
        var replacement = new CaptureState { Input = [37, 41], Output = [59, 61], Bias = 11 };
        if (methodGroup) body = replacement.Execute;
        else wrapper.State = replacement;
        fixture.Gpu.For(0, replacement.Output.Length, body);
        Assert.Equal(new[] { 48, 52 }, replacement.Output);
        Assert.Equal(new[] { 20, 22 }, state.Output);
    }

    [Fact]
    public void StaticVolatileCapturesReadSnapshotsAndRefreshCachedBodies()
    {
        staticInput = [17, 19];
        staticOutput = [41, 43];
        staticBias = 3;
        Action<int> body = StaticBody;
        fixture.Gpu.For(0, staticOutput.Length, body);
        Assert.Equal(new[] { 20, 22 }, staticOutput);
        int[] previousOutput = staticOutput;
        staticInput = [23, 29];
        staticOutput = [47, 53];
        staticBias = 7;
        fixture.Gpu.For<int>(0, staticOutput.Length, body);
        Assert.Equal(new[] { 30, 36 }, staticOutput);
        Assert.Equal(new[] { 20, 22 }, previousOutput);
    }

    [Fact]
    public void StaticVolatileBooleanAndCharacterCapturesRetainTheirMappedValues()
    {
        staticEnabled = true;
        staticMarker = '\uffff';
        int[] output = new int[2];
        Action<int> body = i => output[i] = staticEnabled ? staticMarker + i : -1;
        fixture.Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 65535, 65536 }, output);
        staticEnabled = false;
        staticMarker = '\ud800';
        fixture.Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { -1, -1 }, output);
        staticEnabled = true;
        staticMarker = '\u8000';
        fixture.Gpu.For<int>(0, output.Length, body);
        Assert.Equal(new[] { 32768, 32769 }, output);
    }

    [Fact]
    public void VolatileMappedStructureFieldsRetainTheirPrefixInsideTranslatedMembers()
    {
        var captured = new VolatileValue { Number = 31, Marker = '\uffff' };
        int[] output = new int[2];
        fixture.Gpu.For(0, output.Length, i => output[i] = captured.Project(i));
        Assert.Equal(new[] { 31 + 65535, 32 + 65535 }, output);
        Assert.Equal(31, captured.Number);
        Assert.Equal('\uffff', captured.Marker);
    }

    [Fact]
    public void MappedHelperArgumentsCanBeReassignedAndPassedByAddress()
    {
        Payload[] input =
        [
            new() { Enabled = true, Marker = '\uffff', Bits = UnsignedBits.Low },
            new() { Enabled = false, Marker = '\ud800', Bits = UnsignedBits.High }
        ];
        Payload[] output = new Payload[input.Length];
        fixture.Gpu.For(0, output.Length, i => output[i] = Rewrite(input[i], input[i].Marker, i + 1));
        Payload[] expected = input.Select((value, i) => Rewrite(value, value.Marker, i + 1)).ToArray();
        Assert.Equal(expected.Select(value => value.Enabled), output.Select(value => value.Enabled));
        Assert.Equal(expected.Select(value => value.Marker), output.Select(value => value.Marker));
        Assert.Equal(expected.Select(value => value.Bits), output.Select(value => value.Bits));
        Assert.Equal(expected.Select(value => value.WideBits), output.Select(value => value.WideBits));
        Assert.Equal(new[] { true, false }, input.Select(value => value.Enabled));
        Assert.Equal(new[] { '\uffff', '\ud800' }, input.Select(value => value.Marker));
    }

    private static Payload Rewrite(Payload value, char marker, int amount)
    {
        value = new Payload { Enabled = !value.Enabled, Bits = value.Bits | UnsignedBits.High };
        marker = unchecked((char)(marker + amount));
        WriteMarker(ref value.Marker, ref marker);
        value.WideBits = (long)(uint)value.Bits;
        return value;
    }

    private static void WriteMarker(ref char destination, ref char argument)
    {
        argument = unchecked((char)(argument + 1));
        destination = argument;
    }

    private static void StaticBody(int i) => staticOutput[i] = staticInput[i] + staticBias;

    private sealed class CaptureWrapper { public volatile CaptureState State = null!; }
    private sealed class CaptureState
    {
        public int[] Input = [], Output = [];
        public volatile int Bias;
        public void Execute(int i) => Output[i] = Input[i] + Bias;
    }

    private struct VolatileValue
    {
        public volatile int Number;
        public char Marker;
        public readonly int Project(int amount) => Number + Marker + amount;
    }

    private enum UnsignedBits : uint { Low = 1, High = 0x80000000 }
    private struct Payload
    {
        public bool Enabled;
        public char Marker;
        public UnsignedBits Bits;
        public long WideBits;
    }
}
