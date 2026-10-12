using ILGPU;

namespace ParallelGpu;

/// <summary>Lowers rounding and full-width integer helpers without managed allocation or native wide integers.</summary>
internal static partial class DeviceExtendedMath
{
    public static float RoundSingle(float value, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)RoundIntegralBits(Interop.FloatAsInt(value), 23, 127, 0));

    public static double RoundDouble(double value, DeviceExecutionContext context) =>
        Interop.IntAsFloat(RoundIntegralBits(Interop.FloatAsInt(value), 52, 1023, 0));

    public static float RoundSingle(float value, int digits, DeviceExecutionContext context) =>
        RoundSingle(value, digits, MidpointRounding.ToEven, context);

    public static double RoundDouble(double value, int digits, DeviceExecutionContext context) =>
        RoundDouble(value, digits, MidpointRounding.ToEven, context);

    public static float RoundSingle(float value, MidpointRounding mode, DeviceExecutionContext context)
    {
        if ((uint)mode > 4) { context.Fail(17); return 0; }
        return Interop.IntAsFloat((uint)RoundIntegralBits(Interop.FloatAsInt(value), 23, 127, (int)mode));
    }

    public static double RoundDouble(double value, MidpointRounding mode, DeviceExecutionContext context)
    {
        if ((uint)mode > 4) { context.Fail(17); return 0; }
        return Interop.IntAsFloat(RoundIntegralBits(Interop.FloatAsInt(value), 52, 1023, (int)mode));
    }

    public static float RoundSingle(float value, int digits, MidpointRounding mode, DeviceExecutionContext context)
    {
        if ((uint)digits > 6) { context.Fail(16); return 0; }
        if ((uint)mode > 4) { context.Fail(17); return 0; }
        // Match the framework's decimal scaling range and perform the float operations at float precision.
        float magnitude = Interop.IntAsFloat(Interop.FloatAsInt(value) & 0x7fffffffU);
        if (!(magnitude < 100_000_000f))
            return value;
        float scale = 1;
        for (int i = 0; i < digits; i++)
            scale *= 10;
        float scaled = value * scale;
        float rounded = Interop.IntAsFloat((uint)RoundIntegralBits(Interop.FloatAsInt(scaled), 23, 127, (int)mode));
        return rounded / scale;
    }

    public static double RoundDouble(double value, int digits, MidpointRounding mode, DeviceExecutionContext context)
    {
        if ((uint)digits > 15) { context.Fail(16); return 0; }
        if ((uint)mode > 4) { context.Fail(17); return 0; }
        double magnitude = Interop.IntAsFloat(Interop.FloatAsInt(value) & 0x7fffffffffffffffUL);
        if (!(magnitude < 10_000_000_000_000_000d))
            return value;
        double scale = 1;
        for (int i = 0; i < digits; i++)
            scale *= 10;
        double scaled = value * scale;
        double rounded = Interop.IntAsFloat(RoundIntegralBits(Interop.FloatAsInt(scaled), 52, 1023, (int)mode));
        return rounded / scale;
    }

    public static float TruncateSingle(float value, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)RoundIntegralBits(Interop.FloatAsInt(value), 23, 127, 2));

    public static double TruncateDouble(double value, DeviceExecutionContext context) =>
        Interop.IntAsFloat(RoundIntegralBits(Interop.FloatAsInt(value), 52, 1023, 2));

    /// <summary>Rounds IEEE magnitude bits directly, retaining the sign of zero and all exceptional payloads.</summary>
    /// <remarks>Mode values are ToEven, AwayFromZero, ToZero, ToNegativeInfinity, and ToPositiveInfinity.
    /// Numeric enum values allow the .NET Standard library to lower newer framework enum members.</remarks>
    private static ulong RoundIntegralBits(ulong bits, int fractionBits, int bias, int mode)
    {
        ulong signMask = 1UL << (fractionBits == 52 ? 63 : 31);
        ulong sign = bits & signMask;
        ulong magnitude = bits & (signMask - 1);
        int exponent = (int)(magnitude >> fractionBits) - bias;
        if (exponent >= fractionBits || magnitude == 0)
            return bits;

        bool negative = sign != 0;
        bool increment;
        if (exponent < 0)
        {
            ulong half = (ulong)(bias - 1) << fractionBits;
            increment = mode == 0 ? magnitude > half : mode == 1 ? magnitude >= half :
                mode == 3 ? negative : mode == 4 && !negative;
            return sign | (increment ? (ulong)bias << fractionBits : 0);
        }

        int shift = fractionBits - exponent;
        ulong unit = 1UL << shift;
        ulong discarded = magnitude & (unit - 1);
        if (discarded == 0)
            return bits;
        ulong integral = magnitude & ~(unit - 1);
        increment = mode == 0 ? discarded > (unit >> 1) ||
                (discarded == (unit >> 1) && (integral & unit) != 0) :
            mode == 1 ? discarded >= (unit >> 1) : mode == 3 ? negative : mode == 4 && !negative;
        return sign | (increment ? integral + unit : integral);
    }

    public static int SignSingle(float value, DeviceExecutionContext context)
    {
        uint bits = Interop.FloatAsInt(value);
        uint magnitude = bits & 0x7fffffffU;
        if (magnitude > 0x7f800000U) { context.Fail(18); return 0; }
        return magnitude == 0 ? 0 : (bits & 0x80000000U) != 0 ? -1 : 1;
    }

    public static int SignDouble(double value, DeviceExecutionContext context)
    {
        ulong bits = Interop.FloatAsInt(value);
        ulong magnitude = bits & 0x7fffffffffffffffUL;
        if (magnitude > 0x7ff0000000000000UL) { context.Fail(18); return 0; }
        return magnitude == 0 ? 0 : (bits & 0x8000000000000000UL) != 0 ? -1 : 1;
    }

    public static int SignInt8(sbyte value, DeviceExecutionContext context) => value < 0 ? -1 : value > 0 ? 1 : 0;
    public static int SignInt16(short value, DeviceExecutionContext context) => value < 0 ? -1 : value > 0 ? 1 : 0;
    public static int SignInt32(int value, DeviceExecutionContext context) => value < 0 ? -1 : value > 0 ? 1 : 0;
    public static int SignInt64(long value, DeviceExecutionContext context) => value < 0 ? -1 : value > 0 ? 1 : 0;

    public static long BigMulInt32(int left, int right, DeviceExecutionContext context) => (long)left * right;

    public static ulong BigMulUInt64(ulong left, ulong right, out ulong low, DeviceExecutionContext context)
    {
        // Four 32-bit limb products retain both halves without a UInt128 device representation.
        ulong lower = (ulong)(uint)left * (uint)right;
        ulong crossLeft = (left >> 32) * (uint)right;
        ulong crossRight = (ulong)(uint)left * (right >> 32);
        ulong middle = (lower >> 32) + (uint)crossLeft + (uint)crossRight;
        low = (middle << 32) | (uint)lower;
        return (left >> 32) * (right >> 32) + (crossLeft >> 32) + (crossRight >> 32) + (middle >> 32);
    }

    public static long BigMulInt64(long left, long right, out long low, DeviceExecutionContext context)
    {
        unchecked
        {
            ulong upper = BigMulUInt64((ulong)left, (ulong)right, out ulong lower, context);
            low = (long)lower;
            long signedUpper = (long)upper;
            if (left < 0)
                signedUpper -= right;
            if (right < 0)
                signedUpper -= left;
            return signedUpper;
        }
    }

    public static int DivRemInt32(int left, int right, out int remainder, DeviceExecutionContext context)
    {
        remainder = 0;
        if (right == 0) { context.Fail(11); return 0; }
        if (left == int.MinValue && right == -1) { context.Fail(12); return 0; }
        int quotient = left / right;
        remainder = left - quotient * right;
        return quotient;
    }

    public static long DivRemInt64(long left, long right, out long remainder, DeviceExecutionContext context)
    {
        remainder = 0;
        if (right == 0) { context.Fail(11); return 0; }
        if (left == long.MinValue && right == -1) { context.Fail(12); return 0; }
        long quotient = left / right;
        remainder = left - quotient * right;
        return quotient;
    }

    public static (int Quotient, int Remainder) DivRemInt32(int left, int right, DeviceExecutionContext context)
    {
        int quotient = DivRemInt32(left, right, out int remainder, context);
        return (quotient, remainder);
    }

    public static (long Quotient, long Remainder) DivRemInt64(long left, long right, DeviceExecutionContext context)
    {
        long quotient = DivRemInt64(left, right, out long remainder, context);
        return (quotient, remainder);
    }

    public static (uint Quotient, uint Remainder) DivRemUInt32(uint left, uint right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return default; }
        uint quotient = left / right;
        return (quotient, left - quotient * right);
    }

    public static (ulong Quotient, ulong Remainder) DivRemUInt64(ulong left, ulong right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return default; }
        ulong quotient = left / right;
        return (quotient, left - quotient * right);
    }

    public static (sbyte Quotient, sbyte Remainder) DivRemInt8(sbyte left, sbyte right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return default; }
        // The framework performs promoted division then an unchecked narrow conversion.
        sbyte quotient = unchecked((sbyte)(left / right));
        return (quotient, unchecked((sbyte)(left - quotient * right)));
    }

    public static (byte Quotient, byte Remainder) DivRemUInt8(byte left, byte right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return default; }
        byte quotient = (byte)(left / right);
        return (quotient, (byte)(left - quotient * right));
    }

    public static (short Quotient, short Remainder) DivRemInt16(short left, short right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return default; }
        short quotient = unchecked((short)(left / right));
        return (quotient, unchecked((short)(left - quotient * right)));
    }

    public static (ushort Quotient, ushort Remainder) DivRemUInt16(ushort left, ushort right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return default; }
        ushort quotient = (ushort)(left / right);
        return (quotient, (ushort)(left - quotient * right));
    }
}
