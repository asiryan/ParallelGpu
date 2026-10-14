using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class IntegerPowerDivisionRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void SignedConstantPowerDivisorsTruncateTowardZeroInBodiesHelpersAndConstructors()
    {
        int[] input32 = [-1, -3, -7, int.MinValue, int.MinValue + 1, int.MaxValue, 0, 1];
        long[] input64 = [-1, -3, -7, long.MinValue, long.MinValue + 1, long.MaxValue, 0, 1];
        int[] output32 = new int[input32.Length * 6];
        long[] output64 = new long[input64.Length * 6];
        fixture.Gpu.For(0, input32.Length, i =>
        {
            output32[i * 6] = input32[i] / 2;
            output32[i * 6 + 1] = input32[i] / -2;
            output32[i * 6 + 2] = input32[i] / int.MinValue;
            output32[i * 6 + 3] = DivideSigned32(input32[i]);
            output32[i * 6 + 4] = new DividedValue(input32[i], input64[i]).Signed32;
            output32[i * 6 + 5] = input32[i] / 1;
            output64[i * 6] = input64[i] / 2;
            output64[i * 6 + 1] = input64[i] / -2;
            output64[i * 6 + 2] = input64[i] / long.MinValue;
            output64[i * 6 + 3] = DivideSigned64(input64[i]);
            output64[i * 6 + 4] = new DividedValue(input32[i], input64[i]).Signed64;
            output64[i * 6 + 5] = input64[i] / 1;
        });
        for (int i = 0; i < input32.Length; i++)
        {
            int x = input32[i];
            long y = input64[i];
            Assert.Equal(new[] { x / 2, x / -2, x / int.MinValue, DivideSigned32(x), x / 32, x }, output32.Skip(i * 6).Take(6));
            Assert.Equal(new[] { y / 2, y / -2, y / long.MinValue, DivideSigned64(y), y / 4294967296L, y }, output64.Skip(i * 6).Take(6));
        }
    }

    [Fact]
    public void UnsignedConstantPowerDivisorsRetainHighBitsInBodiesHelpersAndConstructors()
    {
        uint[] input32 = [uint.MaxValue, uint.MaxValue - 1, 0x80000000U, 0x80000001U, int.MaxValue, 0, 1];
        ulong[] input64 = [ulong.MaxValue, ulong.MaxValue - 1, 0x8000000000000000UL, 0x8000000000000001UL, long.MaxValue, 0, 1];
        uint[] output32 = new uint[input32.Length * 5];
        ulong[] output64 = new ulong[input64.Length * 5];
        fixture.Gpu.For(0, input32.Length, i =>
        {
            output32[i * 5] = input32[i] / 2U;
            output32[i * 5 + 1] = input32[i] / 0x80000000U;
            output32[i * 5 + 2] = DivideUnsigned32(input32[i]);
            output32[i * 5 + 3] = new DividedValue(input32[i], input64[i]).Unsigned32;
            output32[i * 5 + 4] = input32[i] / 1U;
            output64[i * 5] = input64[i] / 2UL;
            output64[i * 5 + 1] = input64[i] / 0x8000000000000000UL;
            output64[i * 5 + 2] = DivideUnsigned64(input64[i]);
            output64[i * 5 + 3] = new DividedValue(input32[i], input64[i]).Unsigned64;
            output64[i * 5 + 4] = input64[i] / 1UL;
        });
        for (int i = 0; i < input32.Length; i++)
        {
            uint x = input32[i];
            ulong y = input64[i];
            Assert.Equal(new[] { x / 2U, x / 0x80000000U, DivideUnsigned32(x), x / 32U, x }, output32.Skip(i * 5).Take(5));
            Assert.Equal(new[] { y / 2UL, y / 0x8000000000000000UL, DivideUnsigned64(y), y / 4294967296UL, y }, output64.Skip(i * 5).Take(5));
        }
    }

    [Fact]
    public void DivRemPowerDivisorsMatchBothComponentsAtEveryIntegerWidth()
    {
        sbyte[] signed8 = [-1, -3, sbyte.MinValue, sbyte.MaxValue];
        byte[] unsigned8 = [byte.MaxValue, 128, 1, 0];
        short[] signed16 = [-1, -3, short.MinValue, short.MaxValue];
        ushort[] unsigned16 = [ushort.MaxValue, 32768, 1, 0];
        int[] signed32 = [-1, -3, int.MinValue, int.MaxValue];
        uint[] unsigned32 = [uint.MaxValue, 0x80000000U, 1, 0];
        long[] signed64 = [-1, -3, long.MinValue, long.MaxValue];
        ulong[] unsigned64 = [ulong.MaxValue, 0x8000000000000000UL, 1, 0];
        (sbyte, sbyte)[] result8 = new (sbyte, sbyte)[4];
        (byte, byte)[] unsignedResult8 = new (byte, byte)[4];
        (short, short)[] result16 = new (short, short)[4];
        (ushort, ushort)[] unsignedResult16 = new (ushort, ushort)[4];
        (int, int)[] result32 = new (int, int)[4];
        (uint, uint)[] unsignedResult32 = new (uint, uint)[4];
        (long, long)[] result64 = new (long, long)[4];
        (ulong, ulong)[] unsignedResult64 = new (ulong, ulong)[4];
        int[] quotient32 = new int[4], remainder32 = new int[4];
        long[] quotient64 = new long[4], remainder64 = new long[4];
        fixture.Gpu.For(0, signed8.Length, i =>
        {
            result8[i] = Math.DivRem(signed8[i], (sbyte)2);
            unsignedResult8[i] = Math.DivRem(unsigned8[i], (byte)2);
            result16[i] = Math.DivRem(signed16[i], (short)-2);
            unsignedResult16[i] = Math.DivRem(unsigned16[i], (ushort)2);
            result32[i] = Math.DivRem(signed32[i], -2);
            unsignedResult32[i] = Math.DivRem(unsigned32[i], 2U);
            result64[i] = Math.DivRem(signed64[i], 4294967296L);
            unsignedResult64[i] = Math.DivRem(unsigned64[i], 4294967296UL);
            quotient32[i] = Math.DivRem(signed32[i], 2, out remainder32[i]);
            quotient64[i] = Math.DivRem(signed64[i], -2, out remainder64[i]);
        });
        for (int i = 0; i < signed8.Length; i++)
        {
            Assert.Equal(Math.DivRem(signed8[i], (sbyte)2), result8[i]);
            Assert.Equal(Math.DivRem(unsigned8[i], (byte)2), unsignedResult8[i]);
            Assert.Equal(Math.DivRem(signed16[i], (short)-2), result16[i]);
            Assert.Equal(Math.DivRem(unsigned16[i], (ushort)2), unsignedResult16[i]);
            Assert.Equal(Math.DivRem(signed32[i], -2), result32[i]);
            Assert.Equal(Math.DivRem(unsigned32[i], 2U), unsignedResult32[i]);
            Assert.Equal(Math.DivRem(signed64[i], 4294967296L), result64[i]);
            Assert.Equal(Math.DivRem(unsigned64[i], 4294967296UL), unsignedResult64[i]);
            Assert.Equal(Math.DivRem(signed32[i], 2), (quotient32[i], remainder32[i]));
            Assert.Equal(Math.DivRem(signed64[i], -2), (quotient64[i], remainder64[i]));
        }
    }

    [Fact]
    public void VariableDivisorsPreservePowerAndNonPowerDivision()
    {
        int[] signed32 = [-1, -3, int.MinValue, int.MinValue, int.MaxValue, -7];
        int[] divisor32 = [2, -2, int.MinValue, 1, 3, -3];
        long[] signed64 = [-1, -3, long.MinValue, long.MinValue, long.MaxValue, -7];
        long[] divisor64 = [2, -2, long.MinValue, 1, 3, -3];
        uint[] unsigned32 = [uint.MaxValue, 0x80000001U, 0x80000000U, uint.MaxValue, 1, 0];
        uint[] unsignedDivisor32 = [2U, 0x80000000U, 1U, 3U, 7U, uint.MaxValue];
        ulong[] unsigned64 = [ulong.MaxValue, 0x8000000000000001UL, 0x8000000000000000UL, ulong.MaxValue, 1, 0];
        ulong[] unsignedDivisor64 = [2UL, 0x8000000000000000UL, 1UL, 3UL, 7UL, ulong.MaxValue];
        int[] result32 = new int[signed32.Length];
        long[] result64 = new long[signed32.Length];
        uint[] unsignedResult32 = new uint[signed32.Length];
        ulong[] unsignedResult64 = new ulong[signed32.Length];
        fixture.Gpu.For(0, signed32.Length, i =>
        {
            result32[i] = signed32[i] / divisor32[i];
            result64[i] = signed64[i] / divisor64[i];
            unsignedResult32[i] = unsigned32[i] / unsignedDivisor32[i];
            unsignedResult64[i] = unsigned64[i] / unsignedDivisor64[i];
        });
        Assert.Equal(signed32.Select((value, i) => value / divisor32[i]), result32);
        Assert.Equal(signed64.Select((value, i) => value / divisor64[i]), result64);
        Assert.Equal(unsigned32.Select((value, i) => value / unsignedDivisor32[i]), unsignedResult32);
        Assert.Equal(unsigned64.Select((value, i) => value / unsignedDivisor64[i]), unsignedResult64);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DivisionGuardsStillRollBackAndCachedBodyRecovers(bool wide)
    {
        int[] left32 = [int.MinValue], right32 = [0];
        long[] left64 = [long.MinValue], right64 = [0];
        long[] output = [41];
        Action<int> body = i =>
        {
            output[i] = 99;
            output[i] = wide ? left64[i] / right64[i] : left32[i] / right32[i];
        };
        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        right32[0] = -1;
        right64[0] = -1;
        Assert.Throws<OverflowException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        right32[0] = -2;
        right64[0] = -2;
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(wide ? long.MinValue / -2 : int.MinValue / -2, output[0]);
    }

    private static int DivideSigned32(int value) => value / -8;
    private static long DivideSigned64(long value) => value / -4294967296L;
    private static uint DivideUnsigned32(uint value) => value / 8U;
    private static ulong DivideUnsigned64(ulong value) => value / 8UL;

    private readonly struct DividedValue
    {
        public readonly int Signed32;
        public readonly long Signed64;
        public readonly uint Unsigned32;
        public readonly ulong Unsigned64;
        public DividedValue(int value32, long value64) { Signed32 = value32 / 32; Signed64 = value64 / 4294967296L; }
        public DividedValue(uint value32, ulong value64) { Unsigned32 = value32 / 32U; Unsigned64 = value64 / 4294967296UL; }
    }
}
