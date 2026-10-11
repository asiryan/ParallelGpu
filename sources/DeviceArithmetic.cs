using ILGPU;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Provides guarded integer arithmetic and portable floating-point operations.</summary>
internal static class DeviceArithmetic
{
    internal static bool IsDivision(OpCode code) => code == OpCodes.Div || code == OpCodes.Div_Un ||
        code == OpCodes.Rem || code == OpCodes.Rem_Un;

    internal static bool IsNarrowingConversion(OpCode code) => code == OpCodes.Conv_I1 || code == OpCodes.Conv_U1 ||
        code == OpCodes.Conv_I2 || code == OpCodes.Conv_U2;
    /// <summary>Gets the CIL integer stack width; zero denotes floating point and minus one an unsupported type.</summary>
    internal static int IntegerWidth(Type? type)
    {
        if (type is null)
            return -1;
        if (type.IsEnum)
            type = Enum.GetUnderlyingType(type);
        if (type == typeof(float) || type == typeof(double))
            return 0;
        if (type == typeof(long) || type == typeof(ulong))
            return 64;
        return type == typeof(int) || type == typeof(uint) || type == typeof(short) || type == typeof(ushort) ||
            type == typeof(byte) || type == typeof(sbyte) || type == typeof(char) || type == typeof(bool) ? 32 : -1;
    }

    public static float MinSingle(float left, float right, DeviceExecutionContext context) =>
        float.IsNaN(left) ? left : float.IsNaN(right) ? right : Math.Min(left, right);
    public static float MaxSingle(float left, float right, DeviceExecutionContext context) =>
        float.IsNaN(left) ? left : float.IsNaN(right) ? right : Math.Max(left, right);
    public static double MinDouble(double left, double right, DeviceExecutionContext context) =>
        double.IsNaN(left) ? left : double.IsNaN(right) ? right : Math.Min(left, right);
    public static double MaxDouble(double left, double right, DeviceExecutionContext context) =>
        double.IsNaN(left) ? left : double.IsNaN(right) ? right : Math.Max(left, right);
    public static float CopySignSingle(float value, float sign, DeviceExecutionContext context) =>
        Interop.IntAsFloat((Interop.FloatAsInt(value) & 0x7fffffffU) | (Interop.FloatAsInt(sign) & 0x80000000U));
    public static double CopySignDouble(double value, double sign, DeviceExecutionContext context) =>
        Interop.IntAsFloat((Interop.FloatAsInt(value) & 0x7fffffffffffffffUL) |
            (Interop.FloatAsInt(sign) & 0x8000000000000000UL));

    public static float RemainderSingle(float left, float right, DeviceExecutionContext context)
    {
        uint bits = Interop.FloatAsInt(left);
        uint magnitude = bits & 0x7fffffffU;
        uint divisor = Interop.FloatAsInt(right) & 0x7fffffffU;
        if (divisor == 0 || magnitude >= 0x7f800000U || divisor > 0x7f800000U)
            return float.NaN;
        if (magnitude < divisor)
            return left; // Includes zero dividends and infinite divisors.
        return Interop.IntAsFloat((uint)RemainderBits(magnitude, divisor, 23) | (bits & 0x80000000U));
    }
    public static double RemainderDouble(double left, double right, DeviceExecutionContext context)
    {
        ulong bits = Interop.FloatAsInt(left);
        ulong magnitude = bits & 0x7fffffffffffffffUL;
        ulong divisor = Interop.FloatAsInt(right) & 0x7fffffffffffffffUL;
        if (divisor == 0 || magnitude >= 0x7ff0000000000000UL || divisor > 0x7ff0000000000000UL)
            return double.NaN;
        if (magnitude < divisor)
            return left;
        return Interop.IntAsFloat(RemainderBits(magnitude, divisor, 52) | (bits & 0x8000000000000000UL));
    }

    /// <summary>Computes the exact binary remainder without a rounded or overflowing floating quotient.</summary>
    /// <remarks>Inputs are positive finite encodings with dividend at least divisor. Integer long
    /// division keeps at most one extra significand bit; encoding the result retains subnormal values.</remarks>
    private static ulong RemainderBits(ulong dividend, ulong divisor, int fractionBits)
    {
        if (dividend == divisor)
            return 0;
        ulong implicitBit = 1UL << fractionBits;
        ulong fractionMask = implicitBit - 1;
        int dividendExponent = (int)(dividend >> fractionBits);
        int divisorExponent = (int)(divisor >> fractionBits);
        dividend &= fractionMask;
        divisor &= fractionMask;
        if (dividendExponent == 0)
        {
            dividendExponent = 1;
            while (dividend < implicitBit)
            {
                dividend <<= 1;
                dividendExponent--;
            }
        }
        else
            dividend |= implicitBit;
        if (divisorExponent == 0)
        {
            divisorExponent = 1;
            while (divisor < implicitBit)
            {
                divisor <<= 1;
                divisorExponent--;
            }
        }
        else
            divisor |= implicitBit;
        while (dividendExponent > divisorExponent)
        {
            if (dividend >= divisor)
                dividend -= divisor;
            if (dividend == 0)
                return 0;
            dividend <<= 1;
            dividendExponent--;
        }
        if (dividend >= divisor)
            dividend -= divisor;
        if (dividend == 0)
            return 0;
        while (dividend < implicitBit)
        {
            dividend <<= 1;
            dividendExponent--;
        }
        return dividendExponent > 0
            ? (dividend - implicitBit) | ((ulong)dividendExponent << fractionBits)
            : dividend >> (1 - dividendExponent);
    }
    public static sbyte AbsInt8(sbyte value, DeviceExecutionContext context)
    {
        if (value == sbyte.MinValue) { context.Fail(12); return 0; }
        return (sbyte)(value < 0 ? -value : value);
    }
    public static short AbsInt16(short value, DeviceExecutionContext context)
    {
        if (value == short.MinValue) { context.Fail(12); return 0; }
        return (short)(value < 0 ? -value : value);
    }
    public static int AbsInt32(int value, DeviceExecutionContext context)
    {
        if (value == int.MinValue) { context.Fail(12); return 0; }
        return value < 0 ? -value : value;
    }
    public static long AbsInt64(long value, DeviceExecutionContext context)
    {
        if (value == long.MinValue) { context.Fail(12); return 0; }
        return value < 0 ? -value : value;
    }
    public static int DivideInt32(int left, int right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return 0; }
        if (left == int.MinValue && right == -1) { context.Fail(12); return 0; }
        return left / right;
    }
    public static long DivideInt64(long left, long right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return 0; }
        if (left == long.MinValue && right == -1) { context.Fail(12); return 0; }
        return left / right;
    }
    public static uint DivideUInt32(uint left, uint right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return 0; }
        return left / right;
    }
    public static ulong DivideUInt64(ulong left, ulong right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return 0; }
        return left / right;
    }
    public static int RemainderInt32(int left, int right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return 0; }
        if (left == int.MinValue && right == -1) { context.Fail(12); return 0; }
        return left % right;
    }
    public static long RemainderInt64(long left, long right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return 0; }
        if (left == long.MinValue && right == -1) { context.Fail(12); return 0; }
        return left % right;
    }
    public static uint RemainderUInt32(uint left, uint right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return 0; }
        return left % right;
    }
    public static ulong RemainderUInt64(ulong left, ulong right, DeviceExecutionContext context)
    {
        if (right == 0) { context.Fail(11); return 0; }
        return left % right;
    }
}
