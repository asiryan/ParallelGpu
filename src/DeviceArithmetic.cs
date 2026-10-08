using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Checks integer operations that can otherwise terminate CPU device workers.</summary>
internal static class DeviceArithmetic
{
    internal static bool IsDivision(OpCode code) => code == OpCodes.Div || code == OpCodes.Div_Un ||
        code == OpCodes.Rem || code == OpCodes.Rem_Un;

    internal static bool IsNarrowingConversion(OpCode code) => code == OpCodes.Conv_I1 || code == OpCodes.Conv_U1 ||
        code == OpCodes.Conv_I2 || code == OpCodes.Conv_U2;
    /// <summary>Gets the CIL integer stack width, or zero for floating-point operations.</summary>
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
