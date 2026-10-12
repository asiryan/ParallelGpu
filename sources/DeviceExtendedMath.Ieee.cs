using ILGPU;

namespace ParallelGpu;

/// <summary>Implements exact binary-format operations without relying on host floating-point modes.</summary>
internal static partial class DeviceExtendedMath
{
    public static float BitIncrementSingle(float value, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)IeeeAdjacentBits(Interop.FloatAsInt(value), true, 23, 8));
    public static double BitIncrementDouble(double value, DeviceExecutionContext context) =>
        Interop.IntAsFloat(IeeeAdjacentBits(Interop.FloatAsInt(value), true, 52, 11));
    public static float BitDecrementSingle(float value, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)IeeeAdjacentBits(Interop.FloatAsInt(value), false, 23, 8));
    public static double BitDecrementDouble(double value, DeviceExecutionContext context) =>
        Interop.IntAsFloat(IeeeAdjacentBits(Interop.FloatAsInt(value), false, 52, 11));
    public static int ILogBSingle(float value, DeviceExecutionContext context) =>
        IeeeBinaryLogExponent(Interop.FloatAsInt(value), 23, 8, 127);
    public static int ILogBDouble(double value, DeviceExecutionContext context) =>
        IeeeBinaryLogExponent(Interop.FloatAsInt(value), 52, 11, 1023);
    public static float ScaleBSingle(float value, int exponent, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)IeeeScaleBits(Interop.FloatAsInt(value), exponent, 23, 8, 127));
    public static double ScaleBDouble(double value, int exponent, DeviceExecutionContext context) =>
        Interop.IntAsFloat(IeeeScaleBits(Interop.FloatAsInt(value), exponent, 52, 11, 1023));
    public static float MaxMagnitudeSingle(float left, float right, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)IeeeMagnitudeExtremum(Interop.FloatAsInt(left), Interop.FloatAsInt(right), true, 23, 8));
    public static double MaxMagnitudeDouble(double left, double right, DeviceExecutionContext context) =>
        Interop.IntAsFloat(IeeeMagnitudeExtremum(Interop.FloatAsInt(left), Interop.FloatAsInt(right), true, 52, 11));
    public static float MinMagnitudeSingle(float left, float right, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)IeeeMagnitudeExtremum(Interop.FloatAsInt(left), Interop.FloatAsInt(right), false, 23, 8));
    public static double MinMagnitudeDouble(double left, double right, DeviceExecutionContext context) =>
        Interop.IntAsFloat(IeeeMagnitudeExtremum(Interop.FloatAsInt(left), Interop.FloatAsInt(right), false, 52, 11));
    public static float IEEERemainderSingle(float left, float right, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)IeeeRemainderBits(Interop.FloatAsInt(left), Interop.FloatAsInt(right), 23, 8, 127));
    public static double IEEERemainderDouble(double left, double right, DeviceExecutionContext context) =>
        Interop.IntAsFloat(IeeeRemainderBits(Interop.FloatAsInt(left), Interop.FloatAsInt(right), 52, 11, 1023));
    public static float FusedMultiplyAddSingle(float left, float right, float addend, DeviceExecutionContext context) =>
        Interop.IntAsFloat((uint)IeeeFusedBits(Interop.FloatAsInt(left), Interop.FloatAsInt(right), Interop.FloatAsInt(addend), 23, 8, 127));
    public static double FusedMultiplyAddDouble(double left, double right, double addend, DeviceExecutionContext context) =>
        Interop.IntAsFloat(IeeeFusedBits(Interop.FloatAsInt(left), Interop.FloatAsInt(right), Interop.FloatAsInt(addend), 52, 11, 1023));
    public static float ReciprocalEstimateSingle(float value, DeviceExecutionContext context) => (float)(1.0 / value);
    public static double ReciprocalEstimateDouble(double value, DeviceExecutionContext context) => 1.0 / value;
    public static float ReciprocalSqrtEstimateSingle(float value, DeviceExecutionContext context) => (float)(1.0 / Math.Sqrt(value));
    public static double ReciprocalSqrtEstimateDouble(double value, DeviceExecutionContext context) => 1.0 / Math.Sqrt(value);

    private static ulong IeeeAdjacentBits(ulong bits, bool increment, int fractionBits, int exponentBits)
    {
        ulong sign = 1UL << (fractionBits + exponentBits);
        ulong magnitude = bits & (sign - 1);
        ulong infinity = ((1UL << exponentBits) - 1) << fractionBits;
        if (magnitude >= infinity)
            return magnitude == infinity && ((bits & sign) != 0) == increment ? bits - 1 : bits;
        if (magnitude == 0)
            return increment ? 1 : sign | 1;
        return ((bits & sign) == 0) == increment ? bits + 1 : bits - 1;
    }

    private static int IeeeBinaryLogExponent(ulong bits, int fractionBits, int exponentBits, int bias)
    {
        ulong exponentMask = (1UL << exponentBits) - 1;
        int exponent = (int)((bits >> fractionBits) & exponentMask);
        if ((ulong)exponent == exponentMask)
            return int.MaxValue;
        if (exponent != 0)
            return exponent - bias;
        ulong significand = bits & ((1UL << fractionBits) - 1);
        return significand == 0 ? int.MinValue : IeeeBitLength(significand) - bias - fractionBits;
    }

    private static ulong IeeeScaleBits(ulong bits, int adjustment, int fractionBits, int exponentBits, int bias)
    {
        ulong sign = bits & (1UL << (fractionBits + exponentBits));
        ulong magnitude = bits & ((1UL << (fractionBits + exponentBits)) - 1);
        ulong infinity = ((1UL << exponentBits) - 1) << fractionBits;
        if (magnitude == 0 || magnitude >= infinity)
            return bits;
        IeeeDecodeSignificand(magnitude, fractionBits, bias, out ulong significand, out int unitExponent);
        int length = IeeeBitLength(significand);
        long highestExponent = (long)unitExponent + length - 1 + adjustment;
        if (highestExponent > bias)
            return sign | infinity;
        int minimumUnit = 1 - bias - fractionBits;
        if (highestExponent < minimumUnit - 1)
            return sign;
        if (highestExponent >= 1 - bias)
            return sign | ((ulong)(highestExponent + bias) << fractionBits) |
                ((significand << (fractionBits + 1 - length)) & ((1UL << fractionBits) - 1));
        int shift = (int)((long)minimumUnit - unitExponent - adjustment);
        return sign | IeeeRoundBinary(new IeeeWideBits(0, significand), shift);
    }

    private static ulong IeeeMagnitudeExtremum(ulong left, ulong right, bool maximum, int fractionBits, int exponentBits)
    {
        ulong sign = 1UL << (fractionBits + exponentBits);
        ulong magnitudeLeft = left & (sign - 1);
        ulong magnitudeRight = right & (sign - 1);
        ulong infinity = ((1UL << exponentBits) - 1) << fractionBits;
        if (magnitudeLeft > infinity)
            return left;
        if (magnitudeRight > infinity)
            return right;
        if (magnitudeLeft == magnitudeRight)
            return ((left & sign) == 0) == maximum ? left : right;
        return (magnitudeLeft > magnitudeRight) == maximum ? left : right;
    }

    /// <summary>Retains the integer quotient's parity so exact halfway remainders choose an even quotient.</summary>
    private static ulong IeeeRemainderBits(ulong left, ulong right, int fractionBits, int exponentBits, int bias)
    {
        ulong sign = 1UL << (fractionBits + exponentBits);
        ulong magnitudeLeft = left & (sign - 1);
        ulong magnitudeRight = right & (sign - 1);
        ulong infinity = ((1UL << exponentBits) - 1) << fractionBits;
        if (magnitudeLeft > infinity)
            return left;
        if (magnitudeRight > infinity)
            return right;
        if (magnitudeLeft == infinity || magnitudeRight == 0)
            return infinity | (1UL << (fractionBits - 1));
        if (magnitudeLeft == 0 || magnitudeRight == infinity)
            return left;
        IeeeDecodeSignificand(magnitudeLeft, fractionBits, bias, out ulong numerator, out int numeratorExponent);
        IeeeDecodeSignificand(magnitudeRight, fractionBits, bias, out ulong denominator, out int denominatorExponent);
        IeeeNormalizeSignificand(ref numerator, ref numeratorExponent, fractionBits);
        IeeeNormalizeSignificand(ref denominator, ref denominatorExponent, fractionBits);
        ulong resultSign = left & sign;
        if (numeratorExponent < denominatorExponent)
        {
            if (numeratorExponent < denominatorExponent - 1 || numerator <= denominator)
                return left;
            return resultSign ^ sign | IeeeEncodeSignificand(denominator * 2 - numerator, numeratorExponent, fractionBits, bias);
        }
        while (numeratorExponent > denominatorExponent)
        {
            if (numerator >= denominator)
                numerator -= denominator;
            if (numerator == 0)
                return resultSign;
            numerator <<= 1;
            numeratorExponent--;
        }
        bool oddQuotient = numerator >= denominator;
        if (oddQuotient)
            numerator -= denominator;
        if (numerator * 2 > denominator || numerator * 2 == denominator && oddQuotient)
        {
            numerator = denominator - numerator;
            resultSign ^= sign;
        }
        return resultSign | IeeeEncodeSignificand(numerator, denominatorExponent, fractionBits, bias);
    }

    private static void IeeeDecodeSignificand(ulong magnitude, int fractionBits, int bias, out ulong significand, out int exponent)
    {
        int field = (int)(magnitude >> fractionBits);
        significand = magnitude & ((1UL << fractionBits) - 1);
        if (field != 0)
            significand |= 1UL << fractionBits;
        exponent = (field == 0 ? 1 : field) - bias - fractionBits;
    }

    private static void IeeeNormalizeSignificand(ref ulong significand, ref int exponent, int fractionBits)
    {
        int shift = fractionBits + 1 - IeeeBitLength(significand);
        significand <<= shift;
        exponent -= shift;
    }

    private static ulong IeeeEncodeSignificand(ulong significand, int unitExponent, int fractionBits, int bias)
    {
        if (significand == 0)
            return 0;
        int length = IeeeBitLength(significand);
        int topExponent = unitExponent + length - 1;
        if (topExponent >= 1 - bias)
            return ((ulong)(topExponent + bias) << fractionBits) |
                ((significand << (fractionBits + 1 - length)) & ((1UL << fractionBits) - 1));
        int shift = 1 - bias - fractionBits - unitExponent;
        return shift >= 0 ? significand >> shift : significand << -shift;
    }

    /// <summary>Rounds an exact product and addend once, with enough integer guard bits for cancellation and subnormals.</summary>
    private static ulong IeeeFusedBits(ulong left, ulong right, ulong addend, int fractionBits, int exponentBits, int bias)
    {
        ulong sign = 1UL << (fractionBits + exponentBits);
        ulong mask = sign - 1;
        ulong infinity = ((1UL << exponentBits) - 1) << fractionBits;
        ulong a = left & mask, b = right & mask, c = addend & mask;
        ulong quiet = 1UL << (fractionBits - 1);
        if (a > infinity) return left | quiet;
        if (b > infinity) return right | quiet;
        if (c > infinity) return addend | quiet;
        ulong productSign = (left ^ right) & sign;
        ulong addendSign = addend & sign;
        if (a == infinity || b == infinity)
        {
            if (a == 0 || b == 0 || c == infinity && productSign != addendSign)
                return infinity | quiet;
            return productSign | infinity;
        }
        if (c == infinity)
            return addend;
        if (a == 0 || b == 0)
            return c == 0 ? productSign & addendSign : addend;
        IeeeDecodeSignificand(a, fractionBits, bias, out ulong ma, out int ea);
        IeeeDecodeSignificand(b, fractionBits, bias, out ulong mb, out int eb);
        IeeeDecodeSignificand(c, fractionBits, bias, out ulong mc, out int ec);
        IeeeWideBits product = IeeeMultiplyWide(ma, mb);
        int productLength = IeeeBitLength(product);
        int productTop = ea + eb + productLength - 1;
        int addendLength = IeeeBitLength(mc);
        int addendTop = ec + addendLength - 1;
        int commonTop = mc == 0 || productTop >= addendTop ? productTop : addendTop;
        product = IeeeShiftRightJam(IeeeShiftLeft(product, 127 - productLength), commonTop - productTop);
        IeeeWideBits term = mc == 0 ? default : IeeeShiftRightJam(IeeeShiftLeft(new IeeeWideBits(0, mc), 127 - addendLength), commonTop - addendTop);
        IeeeWideBits sum;
        ulong resultSign;
        if (productSign == addendSign)
        {
            sum = IeeeAddWide(product, term);
            resultSign = productSign;
        }
        else if (IeeeGreaterOrEqual(product, term))
        {
            sum = IeeeSubtractWide(product, term);
            resultSign = productSign;
        }
        else
        {
            sum = IeeeSubtractWide(term, product);
            resultSign = addendSign;
        }
        int length = IeeeBitLength(sum);
        if (length == 0)
            return 0; // Exact cancellation rounds to positive zero.
        int resultTop = commonTop + length - 127;
        if (resultTop < 1 - bias)
            return resultSign | IeeeRoundBinary(sum, 1 - bias - fractionBits - commonTop + 126);
        ulong rounded = IeeeRoundBinary(sum, length - fractionBits - 1);
        if (rounded >= (1UL << (fractionBits + 1)))
        {
            rounded >>= 1;
            resultTop++;
        }
        if (resultTop > bias)
            return resultSign | infinity;
        return resultSign | ((ulong)(resultTop + bias) << fractionBits) | (rounded & ((1UL << fractionBits) - 1));
    }

    private readonly struct IeeeWideBits
    {
        internal readonly ulong High;
        internal readonly ulong Low;
        internal IeeeWideBits(ulong high, ulong low) { High = high; Low = low; }
    }

    private static int IeeeBitLength(ulong value) => value == 0 ? 0 : 64 - IntrinsicMath.LeadingZeroCount(value);
    private static int IeeeBitLength(IeeeWideBits value) => value.High == 0 ? IeeeBitLength(value.Low) : 64 + IeeeBitLength(value.High);
    private static IeeeWideBits IeeeMultiplyWide(ulong left, ulong right)
    {
        ulong low = (ulong)(uint)left * (uint)right;
        ulong middle1 = (left >> 32) * (uint)right;
        ulong middle2 = (ulong)(uint)left * (right >> 32);
        ulong middle = (low >> 32) + (uint)middle1 + (uint)middle2;
        return new((left >> 32) * (right >> 32) + (middle1 >> 32) + (middle2 >> 32) + (middle >> 32),
            (middle << 32) | (uint)low);
    }
    private static bool IeeeGreaterOrEqual(IeeeWideBits left, IeeeWideBits right) =>
        left.High > right.High || left.High == right.High && left.Low >= right.Low;
    private static IeeeWideBits IeeeAddWide(IeeeWideBits left, IeeeWideBits right)
    {
        ulong low = left.Low + right.Low;
        return new(left.High + right.High + (low < left.Low ? 1UL : 0), low);
    }
    private static IeeeWideBits IeeeSubtractWide(IeeeWideBits left, IeeeWideBits right) =>
        new(left.High - right.High - (left.Low < right.Low ? 1UL : 0), left.Low - right.Low);
    private static IeeeWideBits IeeeShiftLeft(IeeeWideBits value, int count) => count == 0 ? value : count >= 64
        ? new(value.Low << (count - 64), 0)
        : new((value.High << count) | (value.Low >> (64 - count)), value.Low << count);
    private static IeeeWideBits IeeeShiftRight(IeeeWideBits value, int count) => count == 0 ? value : count >= 128
        ? default : count >= 64 ? new(0, value.High >> (count - 64))
        : new(value.High >> count, (value.Low >> count) | (value.High << (64 - count)));
    private static bool IeeeAnyBelow(IeeeWideBits value, int count) => count <= 0 ? false : count >= 128
        ? value.High != 0 || value.Low != 0 : count >= 64
            ? value.Low != 0 || count > 64 && (value.High & ((1UL << (count - 64)) - 1)) != 0
            : (value.Low & ((1UL << count) - 1)) != 0;
    private static IeeeWideBits IeeeShiftRightJam(IeeeWideBits value, int count)
    {
        IeeeWideBits shifted = IeeeShiftRight(value, count);
        return new(shifted.High, shifted.Low | (IeeeAnyBelow(value, count) ? 1UL : 0));
    }
    private static ulong IeeeRoundBinary(IeeeWideBits value, int shift)
    {
        if (shift <= 0)
            return value.Low << -shift;
        if (shift > 128)
            return 0;
        ulong result = IeeeShiftRight(value, shift).Low;
        bool halfwayBit = shift <= 64 ? ((value.Low >> (shift - 1)) & 1) != 0 :
            ((value.High >> (shift - 65)) & 1) != 0;
        return halfwayBit && (IeeeAnyBelow(value, shift - 1) || (result & 1) != 0) ? result + 1 : result;
    }
}
