using ILGPU;
using ILGPU.Algorithms;

namespace ParallelGpu;

/// <summary>Provides device implementations of supported framework math.</summary>
/// <remarks>Transcendental results are approximate; Exp and Pow can differ at underflow and overflow
/// boundaries. Floating-point evaluation also depends on ILGPU backend optimizations.</remarks>
internal static partial class DeviceExtendedMath
{
    private const double Ln2High = 0.69314718036912381649017333984375;
    private const double Ln2Low = 1.90821492927058770002e-10;
    private const double InverseLn2 = 1.44269504088896340736;
    private const ulong MagnitudeMask = 0x7fffffffffffffffUL;
    private const ulong InfinityBits = 0x7ff0000000000000UL;

    public static float SinSingle(float value, DeviceExecutionContext context) => (float)Sin(value);
    public static double SinDouble(double value, DeviceExecutionContext context) => Sin(value);
    public static float CosSingle(float value, DeviceExecutionContext context) => (float)Cos(value);
    public static double CosDouble(double value, DeviceExecutionContext context) => Cos(value);
    public static float ExpSingle(float value, DeviceExecutionContext context) => (float)Exp(value);
    public static double ExpDouble(double value, DeviceExecutionContext context) => Exp(value);
    public static float PowSingle(float value, float power, DeviceExecutionContext context) => (float)Pow(value, power);
    public static double PowDouble(double value, double power, DeviceExecutionContext context) => Pow(value, power);
    public static float LogSingle(float value, DeviceExecutionContext context) => (float)Log(value);
    public static double LogDouble(double value, DeviceExecutionContext context) => Log(value);
    public static float LogSingle(float value, float newBase, DeviceExecutionContext context) => (float)Log(value, newBase);
    public static double LogDouble(double value, double newBase, DeviceExecutionContext context) => Log(value, newBase);

    private static double Sin(double value)
    {
        ulong bits = (ulong)Interop.FloatAsInt(value);
        ulong magnitude = bits & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        if (magnitude == InfinityBits)
            return double.NaN;
        double positive = Interop.IntAsFloat(magnitude);
        // CORDIC has a small absolute error at zero. Preserve subnormals and signed zero.
        if (positive < 1e-8)
            return value;
        double reduced = ReduceAngle(positive, out int quadrant);
        double result = quadrant switch
        {
            0 => XMath.Sin(reduced),
            1 => XMath.Cos(reduced),
            2 => -XMath.Sin(reduced),
            _ => -XMath.Cos(reduced),
        };
        return (bits >> 63) != 0 ? -result : result;
    }

    private static double Cos(double value)
    {
        ulong magnitude = (ulong)Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        if (magnitude == InfinityBits)
            return double.NaN;
        double positive = Interop.IntAsFloat(magnitude);
        if (positive < 1e-8)
            return 1;
        double reduced = ReduceAngle(positive, out int quadrant);
        return quadrant switch
        {
            0 => XMath.Cos(reduced),
            1 => -XMath.Sin(reduced),
            2 => -XMath.Cos(reduced),
            _ => XMath.Sin(reduced),
        };
    }

    /// <summary>Reduces a nonnegative finite double to a remainder in [-pi/4, pi/4] and its quadrant in bounded work.</summary>
    private static double ReduceAngle(double positive, out int quadrant)
    {
        quadrant = 0;
        if (positive <= 0.7853981633974483)
            return positive;
        ulong bits = (ulong)Interop.FloatAsInt(positive);
        int exponent = (int)(bits >> 52) - 1023;
        ulong significand = (bits & 0x000fffffffffffffUL) | 0x0010000000000000UL;
        uint low = (uint)significand;
        uint high = (uint)(significand >> 32);
        // floor((2/pi)*2^1280) leaves over 250 guard bits even at Double.MaxValue.
        int shift = 1280 + 52 - exponent;
        int first = (shift - 64) >> 5;
        int offset = (shift - 64) & 31;
        uint word0 = 0, word1 = 0, word2 = 0, word3 = 0;
        ulong carry = 0;
        uint previous = 0;
        for (int i = 0; i <= first + 3; i++)
        {
            uint current = TwoOverPiWord(i);
            ulong a = (ulong)current * low;
            ulong b = (ulong)previous * high;
            ulong sum = (uint)a + (ulong)(uint)b + carry;
            uint word = (uint)sum;
            carry = (a >> 32) + (b >> 32) + (sum >> 32);
            previous = current;
            if (i == first) word0 = word;
            if (i == first + 1) word1 = word;
            if (i == first + 2) word2 = word;
            if (i == first + 3) word3 = word;
        }
        ulong fraction;
        if (offset == 0)
        {
            fraction = word0 | ((ulong)word1 << 32);
            quadrant = (int)(word2 & 3);
        }
        else
        {
            fraction = ((ulong)word0 >> offset) | ((ulong)word1 << (32 - offset)) |
                ((ulong)word2 << (64 - offset));
            quadrant = (int)(((word2 >> offset) | (word3 << (32 - offset))) & 3);
        }
        double remainder;
        if ((fraction >> 63) != 0)
        {
            quadrant = (quadrant + 1) & 3;
            remainder = -(double)unchecked(0UL - fraction) * 5.42101086242752217004e-20;
        }
        else
            remainder = (double)fraction * 5.42101086242752217004e-20;
        return remainder * 1.57079632679489655800 + remainder * 6.12323399573676603587e-17;
    }

    private static uint TwoOverPiWord(int index) => index switch
    {
        0 => 0x9af4361du, 1 => 0xf0cfbc20u, 2 => 0xfc7b6babu, 3 => 0x56033046u,
        4 => 0x1f8d5d08u, 5 => 0x6bfb5fb1u, 6 => 0x8a5292eau, 7 => 0x3d0739f7u,
        8 => 0xebe5f17bu, 9 => 0x7527bac7u, 10 => 0x9e5fea2du, 11 => 0x4f463f66u,
        12 => 0x27cb09b7u, 13 => 0x6d367ecfu, 14 => 0x5a0a6d1fu, 15 => 0xef2f118bu,
        16 => 0xde05980fu, 17 => 0x1ff897ffu, 18 => 0xbdf9283bu, 19 => 0x9c845f8bu,
        20 => 0x835339f4u, 21 => 0x3991d639u, 22 => 0xb45f7e41u, 23 => 0xe99c7026u,
        24 => 0x2ebb4484u, 25 => 0xe88235f5u, 26 => 0xb129a73eu, 27 => 0xfe1deb1cu,
        28 => 0x09d1921cu, 29 => 0x06492eeau, 30 => 0x424dd2e0u, 31 => 0xb7246e3au,
        32 => 0xdebbc561u, 33 => 0xfe5163abu, 34 => 0x3c439041u, 35 => 0xdb629599u,
        36 => 0xf534ddc0u, 37 => 0xfc2757d1u, 38 => 0x4e441529u, 39 => 0xa2f9836eu,
        _ => 0,
    };

    private static double Log(double value)
    {
        ulong bits = (ulong)Interop.FloatAsInt(value);
        ulong magnitude = bits & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        if (magnitude == 0)
            return double.NegativeInfinity;
        if ((bits >> 63) != 0)
            return double.NaN;
        if (magnitude == InfinityBits)
            return value;
        int adjustment = 0;
        if (magnitude < 0x0010000000000000UL)
        {
            value *= 18014398509481984.0;
            bits = (ulong)Interop.FloatAsInt(value);
            adjustment = -54;
        }
        int exponent = (int)(bits >> 52) - 1023 + adjustment;
        double mantissa = Interop.IntAsFloat((bits & 0x000fffffffffffffUL) | 0x3ff0000000000000UL);
        if (mantissa > 1.4142135623730951)
        {
            mantissa *= 0.5;
            exponent++;
        }
        double z = (mantissa - 1) / (mantissa + 1);
        double square = z * z;
        // log(m) = 2*atanh((m-1)/(m+1)); |z| <= 0.172 after normalization.
        double series = 1.0 / 31;
        for (int denominator = 29; denominator >= 3; denominator -= 2)
            series = 1.0 / denominator + square * series;
        double logarithm = 2 * z * (1 + square * series);
        return exponent * Ln2High + (logarithm + exponent * Ln2Low);
    }

    private static double Log(double value, double newBase)
    {
        if (((ulong)Interop.FloatAsInt(value) & MagnitudeMask) > InfinityBits)
            return value;
        if (((ulong)Interop.FloatAsInt(newBase) & MagnitudeMask) > InfinityBits)
            return newBase;
        if (newBase == 1 || (value != 1 && (newBase == 0 || newBase == double.PositiveInfinity)))
            return double.NaN;
        return Log(value) / Log(newBase);
    }

    private static double Pow(double value, double power)
    {
        if (power == 0 || value == 1)
            return 1;
        ulong valueBits = (ulong)Interop.FloatAsInt(value);
        ulong magnitudeBits = valueBits & MagnitudeMask;
        ulong powerBits = (ulong)Interop.FloatAsInt(power) & MagnitudeMask;
        if (magnitudeBits > InfinityBits || powerBits > InfinityBits)
            return double.NaN;
        double magnitude = Interop.IntAsFloat(magnitudeBits);
        if (powerBits == InfinityBits)
        {
            if (magnitude == 1)
                return 1;
            return (magnitude > 1) == (power > 0) ? double.PositiveInfinity : 0;
        }
        bool odd = IsOddInteger(power);
        if (magnitudeBits == 0 || magnitudeBits == InfinityBits)
        {
            double exceptional = (magnitudeBits == 0) == (power > 0) ? 0 : double.PositiveInfinity;
            return odd && (valueBits >> 63) != 0 ? Negate(exceptional) : exceptional;
        }
        if ((valueBits >> 63) != 0 && power != Math.Floor(power))
            return double.NaN;
        if (power == 1)
            return value;
        if (power == -1)
            return 1 / value;
        if (power == 2)
            return value * value;
        double result;
        if (TryBinaryExponent(magnitudeBits, out int binaryExponent) &&
            binaryExponent * power == Math.Floor(binaryExponent * power))
            result = BinaryPower(binaryExponent * power);
        else
            result = Exp(power * Log(magnitude));
        return odd && (valueBits >> 63) != 0 ? Negate(result) : result;
    }

    private static bool TryBinaryExponent(ulong magnitudeBits, out int exponent)
    {
        exponent = (int)(magnitudeBits >> 52) - 1023;
        if (magnitudeBits >= 0x0010000000000000UL)
            return (magnitudeBits & 0x000fffffffffffffUL) == 0;
        if ((magnitudeBits & (magnitudeBits - 1)) != 0)
            return false;
        exponent = -1074;
        while (magnitudeBits > 1)
        {
            magnitudeBits >>= 1;
            exponent++;
        }
        return true;
    }

    private static double BinaryPower(double exponent)
    {
        if (exponent >= 1024)
            return double.PositiveInfinity;
        if (exponent < -1074)
            return 0;
        int integer = (int)exponent;
        return integer < -1022 ? Interop.IntAsFloat(1UL << (integer + 1074)) :
            Interop.IntAsFloat((ulong)(integer + 1023) << 52);
    }

    private static bool IsOddInteger(double value) => value > -9007199254740992.0 &&
        value < 9007199254740992.0 && value == Math.Floor(value) && (((long)value & 1) != 0);

    private static double Negate(double value) =>
        Interop.IntAsFloat(Interop.FloatAsInt(value) ^ 0x8000000000000000UL);

    private static double Exp(double value)
    {
        // Handle NaN and infinities before converting the range-reduction exponent to an integer.
        ulong magnitude = (ulong)Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        if (magnitude == InfinityBits)
            return value > 0 ? double.PositiveInfinity : 0;
        if (value > 709.782712893384)
            return double.PositiveInfinity;
        if (value < -745.1332191019412)
            return 0;
        int exponent = (int)Math.Floor(value * InverseLn2 + 0.5);
        double reduced = (value - exponent * Ln2High) - exponent * Ln2Low;
        double series = 1.0;
        // Nested Taylor form for exp(r), with |r| <= log(2)/2.
        for (int denominator = 18; denominator >= 1; denominator--)
            series = 1 + reduced * series / denominator;
        if (exponent > 1023)
            // The bounded input permits only exponent 1024 here, with series below one.
            // Adjust its encoding instead of multiplying constants that a backend can fold to infinity.
            return Interop.IntAsFloat(Interop.FloatAsInt(series) + ((ulong)exponent << 52));
        if (exponent < -1022)
        {
            double scale = Interop.IntAsFloat((ulong)(exponent + 1022 + 1023) << 52);
            return (series * scale) * Interop.IntAsFloat(0x0010000000000000UL);
        }
        return series * Interop.IntAsFloat((ulong)(exponent + 1023) << 52);
    }
}
