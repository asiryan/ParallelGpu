using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AdditionalArithmeticTranslationRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void NarrowArrayLoadsMergeWithIntegerConstantsWithoutTruncation()
    {
        byte[] bytes = [byte.MaxValue, 128, 1, 0];
        sbyte[] signedBytes = [sbyte.MinValue, -1, 127, 0];
        short[] shorts = [short.MinValue, -1, short.MaxValue, 0];
        ushort[] unsignedShorts = [ushort.MaxValue, 32768, 1, 0];
        int[] byteResult = new int[bytes.Length], signedByteResult = new int[bytes.Length];
        int[] shortResult = new int[bytes.Length], unsignedShortResult = new int[bytes.Length];

        fixture.Gpu.For(0, bytes.Length, i =>
        {
            byteResult[i] = (i % 2 == 0 ? bytes[i] : 1000) / 2;
            signedByteResult[i] = (i % 2 == 0 ? signedBytes[i] : 1000) / 2;
            shortResult[i] = (i % 2 == 0 ? shorts[i] : 100000) / 2;
            unsignedShortResult[i] = (i % 2 == 0 ? unsignedShorts[i] : 100000) / 2;
        });

        Assert.Equal(bytes.Select((value, i) => (i % 2 == 0 ? value : 1000) / 2), byteResult);
        Assert.Equal(signedBytes.Select((value, i) => (i % 2 == 0 ? value : 1000) / 2), signedByteResult);
        Assert.Equal(shorts.Select((value, i) => (i % 2 == 0 ? value : 100000) / 2), shortResult);
        Assert.Equal(unsignedShorts.Select((value, i) => (i % 2 == 0 ? value : 100000) / 2), unsignedShortResult);
    }

    [Fact]
    public void NarrowHelperReturnsMergeWithIntegerConstantsWithoutTruncation()
    {
        byte[] bytes = [byte.MaxValue, 128, 1, 0];
        sbyte[] signedBytes = [sbyte.MinValue, -1, 127, 0];
        short[] shorts = [short.MinValue, -1, short.MaxValue, 0];
        ushort[] unsignedShorts = [ushort.MaxValue, 32768, 1, 0];
        int[] byteResult = new int[bytes.Length], signedByteResult = new int[bytes.Length];
        int[] shortResult = new int[bytes.Length], unsignedShortResult = new int[bytes.Length];

        fixture.Gpu.For(0, bytes.Length, i =>
        {
            byteResult[i] = (i % 2 == 0 ? IdentityByte(bytes[i]) : 1000) / 2;
            signedByteResult[i] = (i % 2 == 0 ? IdentitySignedByte(signedBytes[i]) : 1000) / 2;
            shortResult[i] = (i % 2 == 0 ? IdentityShort(shorts[i]) : 100000) / 2;
            unsignedShortResult[i] = (i % 2 == 0 ? IdentityUnsignedShort(unsignedShorts[i]) : 100000) / 2;
        });

        Assert.Equal(bytes.Select((value, i) => (i % 2 == 0 ? value : 1000) / 2), byteResult);
        Assert.Equal(signedBytes.Select((value, i) => (i % 2 == 0 ? value : 1000) / 2), signedByteResult);
        Assert.Equal(shorts.Select((value, i) => (i % 2 == 0 ? value : 100000) / 2), shortResult);
        Assert.Equal(unsignedShorts.Select((value, i) => (i % 2 == 0 ? value : 100000) / 2), unsignedShortResult);
    }

    [Fact]
    public void NarrowHelperParametersRetainTheirCilStackWidthAcrossArithmeticBranches()
    {
        byte[] bytes = [byte.MaxValue, 128, 1, 0];
        sbyte[] signedBytes = [sbyte.MinValue, -1, 127, 0];
        short[] shorts = [short.MinValue, -1, short.MaxValue, 0];
        ushort[] unsignedShorts = [ushort.MaxValue, 32768, 1, 0];
        int[] byteResult = new int[bytes.Length], signedByteResult = new int[bytes.Length];
        int[] shortResult = new int[bytes.Length], unsignedShortResult = new int[bytes.Length];

        fixture.Gpu.For(0, bytes.Length, i =>
        {
            byteResult[i] = SelectByte(bytes[i], i % 2 == 0);
            signedByteResult[i] = SelectSignedByte(signedBytes[i], i % 2 == 0);
            shortResult[i] = SelectShort(shorts[i], i % 2 == 0);
            unsignedShortResult[i] = SelectUnsignedShort(unsignedShorts[i], i % 2 == 0);
        });

        Assert.Equal(bytes.Select((value, i) => SelectByte(value, i % 2 == 0)), byteResult);
        Assert.Equal(signedBytes.Select((value, i) => SelectSignedByte(value, i % 2 == 0)), signedByteResult);
        Assert.Equal(shorts.Select((value, i) => SelectShort(value, i % 2 == 0)), shortResult);
        Assert.Equal(unsignedShorts.Select((value, i) => SelectUnsignedShort(value, i % 2 == 0)), unsignedShortResult);
    }

    [Fact]
    public void BooleanAndCharacterHelperReturnsRetainTheirCilWidthsAcrossBranches()
    {
        bool[] choices = [true, false, false, true];
        char[] characters = ['\uffff', '\u8000', '\ud800', 'A'];
        bool[] booleanResult = new bool[choices.Length];
        int[] characterResult = new int[choices.Length];
        fixture.Gpu.For(0, choices.Length, i =>
        {
            booleanResult[i] = i % 2 == 0 ? IdentityBoolean(choices[i]) : true;
            characterResult[i] = (i % 2 == 0 ? IdentityCharacter(characters[i]) : 100000) / 2;
        });
        Assert.Equal(choices.Select((value, i) => i % 2 == 0 ? value : true), booleanResult);
        Assert.Equal(characters.Select((value, i) => (i % 2 == 0 ? value : 100000) / 2), characterResult);
    }

    [Fact]
    public void NarrowBranchesInsideHelpersStillDeferDivisionByZeroAndRecover()
    {
        byte[] input = [byte.MaxValue, 128];
        int[] divisors = [2, 0], output = [41, 43];
        Action<int> body = i =>
        {
            output[i] = 99;
            output[i] = SelectAndDivide(input[i], i % 2 == 0, divisors[i]);
        };
        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, input.Length, body));
        Assert.Equal(new[] { 41, 43 }, output);
        divisors[1] = -3;
        fixture.Gpu.For(0, input.Length, body);
        Assert.Equal(input.Select((value, i) => SelectAndDivide(value, i % 2 == 0, divisors[i])), output);
    }

    [Fact]
    public void NarrowRefReturnLoadsPreserveSignAcrossConditionalArithmetic()
    {
        sbyte[] signedBytes = [sbyte.MinValue, -1, sbyte.MaxValue, 0];
        ushort[] unsignedShorts = [ushort.MaxValue, 32768, 1, 0];
        int[] signedResult = new int[signedBytes.Length], unsignedResult = new int[signedBytes.Length];
        fixture.Gpu.For(0, signedBytes.Length, i =>
        {
            signedResult[i] = (i % 2 == 0 ? Element(signedBytes, i) : 1000) % 7;
            unsignedResult[i] = (i % 2 == 0 ? Element(unsignedShorts, i) : 100000) / 3;
        });
        Assert.Equal(signedBytes.Select((value, i) => (i % 2 == 0 ? value : 1000) % 7), signedResult);
        Assert.Equal(unsignedShorts.Select((value, i) => (i % 2 == 0 ? value : 100000) / 3), unsignedResult);
    }

    [Fact]
    public void RefReturnedEnumsKeepTheirUnderlyingWidthsAcrossHelpersAndArithmetic()
    {
        SignedCode[] signed = [(SignedCode)sbyte.MinValue, (SignedCode)(-1), (SignedCode)sbyte.MaxValue, 0];
        UnsignedCode[] unsigned = [(UnsignedCode)ulong.MaxValue, (UnsignedCode)(1UL << 63), (UnsignedCode)7, 0];
        int[] signedResult = new int[signed.Length];
        ulong[] unsignedResult = new ulong[signed.Length];
        fixture.Gpu.For(0, signed.Length, i =>
        {
            signedResult[i] = (i % 2 == 0 ? (sbyte)Element(signed, i) : 1000) / 3;
            unsignedResult[i] = (i % 2 == 0 ? (ulong)Element(unsigned, i) : 1UL << 63) % 7;
        });
        Assert.Equal(signed.Select((value, i) => (i % 2 == 0 ? (sbyte)value : 1000) / 3), signedResult);
        Assert.Equal(unsigned.Select((value, i) => (i % 2 == 0 ? (ulong)value : 1UL << 63) % 7), unsignedResult);
    }

    [Fact]
    public void NestedReceiverAliasesCanReadCaptureWhileMutatingIndependentCopyAliases()
    {
        var captured = new Receiver { Child = new NestedReceiver { Number = 17 } };
        int[] result = new int[4];
        fixture.Gpu.For(0, result.Length, i => result[i] = captured.TransformNestedCopy(i + 1));
        Assert.Equal(Enumerable.Range(0, result.Length).Select(i => captured.TransformNestedCopy(i + 1)), result);
        Assert.Equal(17, captured.Child.Number);
    }

    private static byte IdentityByte(byte value) => value;
    private static sbyte IdentitySignedByte(sbyte value) => value;
    private static short IdentityShort(short value) => value;
    private static ushort IdentityUnsignedShort(ushort value) => value;
    private static bool IdentityBoolean(bool value) => value;
    private static char IdentityCharacter(char value) => value;
    private static int SelectByte(byte value, bool choose) => (choose ? value : 1000) / 2;
    private static int SelectSignedByte(sbyte value, bool choose) => (choose ? value : 1000) / 2;
    private static int SelectShort(short value, bool choose) => (choose ? value : 100000) / 2;
    private static int SelectUnsignedShort(ushort value, bool choose) => (choose ? value : 100000) / 2;
    private static int SelectAndDivide(byte value, bool choose, int divisor) => (choose ? value : 1000) / divisor;
    private static ref T Element<T>(T[] values, int index) => ref values[index];
    private enum SignedCode : sbyte { }
    private enum UnsignedCode : ulong { }

    private struct Receiver
    {
        public NestedReceiver Child;
        public int TransformNestedCopy(int count)
        {
            ref NestedReceiver source = ref Child;
            var copy = source;
            ref int selected = ref copy.Number;
            for (int i = 0; i < count; i++)
                selected += i + source.Number;
            return selected;
        }
    }

    private struct NestedReceiver
    {
        public int Number;
    }
}
