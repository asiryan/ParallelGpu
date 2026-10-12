using ILGPU;

namespace ParallelGpu;

/// <summary>Provides bounded-range trigonometric, hyperbolic, root, and logarithm lowering.</summary>
/// <remarks>Results are approximate; range reduction and polynomial evaluation use double intermediates.</remarks>
internal static partial class DeviceExtendedMath
{
    private const double Pi = 3.141592653589793238462643383279502884;
    private const double HalfPi = 1.570796326794896619231321691639751442;
    private const double QuarterPi = 0.785398163397448309615660845819875721;
    private const double InverseLn10 = 0.434294481903251827651128918916605082;

    public static float TanSingle(float value, DeviceExecutionContext context) => (float)Tan(value);
    public static double TanDouble(double value, DeviceExecutionContext context) => Tan(value);
    public static float AsinSingle(float value, DeviceExecutionContext context) => (float)Asin(value);
    public static double AsinDouble(double value, DeviceExecutionContext context) => Asin(value);
    public static float AcosSingle(float value, DeviceExecutionContext context) => (float)Acos(value);
    public static double AcosDouble(double value, DeviceExecutionContext context) => Acos(value);
    public static float AtanSingle(float value, DeviceExecutionContext context) => (float)Atan(value);
    public static double AtanDouble(double value, DeviceExecutionContext context) => Atan(value);
    public static float Atan2Single(float y, float x, DeviceExecutionContext context) => (float)Atan2(y, x);
    public static double Atan2Double(double y, double x, DeviceExecutionContext context) => Atan2(y, x);
    public static float SinhSingle(float value, DeviceExecutionContext context) => (float)Sinh(value);
    public static double SinhDouble(double value, DeviceExecutionContext context) => Sinh(value);
    public static float CoshSingle(float value, DeviceExecutionContext context) => (float)Cosh(value);
    public static double CoshDouble(double value, DeviceExecutionContext context) => Cosh(value);
    public static float TanhSingle(float value, DeviceExecutionContext context) => (float)Tanh(value);
    public static double TanhDouble(double value, DeviceExecutionContext context) => Tanh(value);
    public static float AsinhSingle(float value, DeviceExecutionContext context) => (float)Asinh(value);
    public static double AsinhDouble(double value, DeviceExecutionContext context) => Asinh(value);
    public static float AcoshSingle(float value, DeviceExecutionContext context) => (float)Acosh(value);
    public static double AcoshDouble(double value, DeviceExecutionContext context) => Acosh(value);
    public static float AtanhSingle(float value, DeviceExecutionContext context) => (float)Atanh(value);
    public static double AtanhDouble(double value, DeviceExecutionContext context) => Atanh(value);
    public static float CbrtSingle(float value, DeviceExecutionContext context) => (float)Cbrt(value);
    public static double CbrtDouble(double value, DeviceExecutionContext context) => Cbrt(value);
    public static float Log2Single(float value, DeviceExecutionContext context) => (float)LogBaseTwo(value);
    public static double Log2Double(double value, DeviceExecutionContext context) => LogBaseTwo(value);
    public static float Log10Single(float value, DeviceExecutionContext context) => (float)(Log(value) * InverseLn10);
    public static double Log10Double(double value, DeviceExecutionContext context) => Log(value) * InverseLn10;
    public static (float Sin, float Cos) SinCosSingle(float value, DeviceExecutionContext context) => ((float)Sin(value), (float)Cos(value));
    public static (double Sin, double Cos) SinCosDouble(double value, DeviceExecutionContext context) => (Sin(value), Cos(value));

    /// <summary>Copies a sign bit without losing signed zero or a NaN's magnitude bits.</summary>
    private static double SignMagnitude(double magnitude, double sign) =>
        Interop.IntAsFloat((Interop.FloatAsInt(magnitude) & MagnitudeMask) |
            (Interop.FloatAsInt(sign) & ~MagnitudeMask));

    /// <summary>Computes tangent after accurate quadrant reduction, avoiding division by an inaccurate near-zero cosine.</summary>
    private static double Tan(double value)
    {
        ulong magnitude = Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        if (magnitude == InfinityBits)
            return double.NaN;
        double positive = Interop.IntAsFloat(magnitude);
        if (positive < 1e-8)
            return value;
        double reduced = ReduceTanAngle(positive, out int quadrant);
        double tangent = TanReduced(reduced);
        if ((quadrant & 1) != 0)
            tangent = -1 / tangent;
        return (Interop.FloatAsInt(value) >> 63) != 0 ? -tangent : tangent;
    }

    /// <summary>Evaluates sine and cosine on [-pi/4, pi/4] without CORDIC's absolute error near zero.</summary>
    private static double TanReduced(double reduced)
    {
        double square = reduced * reduced;
        double sineTerm = reduced;
        double cosineTerm = 1;
        double sine = reduced;
        double cosine = 1;
        // Through degrees 21 and 20, the omitted terms are below 6e-24 on the reduced interval.
        for (int n = 2; n <= 20; n += 2)
        {
            cosineTerm *= -square / ((n - 1) * n);
            sineTerm *= -square / (n * (n + 1));
            cosine += cosineTerm;
            sine += sineTerm;
        }
        return sine / cosine;
    }

    /// <summary>Retains 128 fractional reduction bits so tangent remains accurate near odd multiples of pi/2.</summary>
    private static double ReduceTanAngle(double positive, out int quadrant)
    {
        quadrant = 0;
        if (positive <= QuarterPi)
            return positive;
        ulong bits = Interop.FloatAsInt(positive);
        int exponent = (int)(bits >> 52) - 1023;
        ulong significand = (bits & 0x000fffffffffffffUL) | 0x0010000000000000UL;
        uint low = (uint)significand;
        uint high = (uint)(significand >> 32);
        int shift = 1280 + 52 - exponent;
        int first = (shift - 128) >> 5;
        int offset = (shift - 128) & 31;
        uint word0 = 0, word1 = 0, word2 = 0, word3 = 0, word4 = 0, word5 = 0;
        ulong carry = 0;
        uint previous = 0;
        for (int i = 0; i <= first + 5; i++)
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
            if (i == first + 4) word4 = word;
            if (i == first + 5) word5 = word;
        }
        ulong lower, upper;
        if (offset == 0)
        {
            lower = word0 | ((ulong)word1 << 32);
            upper = word2 | ((ulong)word3 << 32);
            quadrant = (int)(word4 & 3);
        }
        else
        {
            lower = ((ulong)word0 >> offset) | ((ulong)word1 << (32 - offset)) | ((ulong)word2 << (64 - offset));
            upper = ((ulong)word2 >> offset) | ((ulong)word3 << (32 - offset)) | ((ulong)word4 << (64 - offset));
            quadrant = (int)(((word4 >> offset) | (word5 << (32 - offset))) & 3);
        }
        bool negative = (upper >> 63) != 0;
        if (negative)
        {
            quadrant = (quadrant + 1) & 3;
            upper = unchecked(~upper + (lower == 0 ? 1UL : 0UL));
            lower = unchecked(0UL - lower);
        }
        double fraction = (double)upper * 5.42101086242752217004e-20 + (double)lower * 2.93873587705571876992e-39;
        double remainder = fraction * HalfPi + fraction * 6.12323399573676603587e-17;
        return negative ? -remainder : remainder;
    }

    /// <summary>Uses a bounded alternating series after reciprocal and pi/4 range reductions.</summary>
    private static double Atan(double value)
    {
        ulong magnitude = Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        if (magnitude == InfinityBits)
            return SignMagnitude(HalfPi, value);
        double positive = Interop.IntAsFloat(magnitude);
        if (positive < 1e-16)
            return value;
        bool reciprocal = positive > 1;
        if (reciprocal)
            positive = 1 / positive;
        bool shifted = positive > 0.41421356237309504880;
        double reduced = shifted ? (positive - 1) / (positive + 1) : positive;
        double square = reduced * reduced;
        double series = 1.0 / 49;
        for (int denominator = 47; denominator >= 3; denominator -= 2)
            series = 1.0 / denominator - square * series;
        double result = reduced * (1 - square * series);
        if (shifted)
            result += QuarterPi;
        if (reciprocal)
            result = HalfPi - result;
        return SignMagnitude(result, value);
    }

    /// <summary>Handles all signed-zero and infinite quadrants before reducing the finite ratio.</summary>
    private static double Atan2(double y, double x)
    {
        ulong yMagnitude = Interop.FloatAsInt(y) & MagnitudeMask;
        ulong xMagnitude = Interop.FloatAsInt(x) & MagnitudeMask;
        if (yMagnitude > InfinityBits)
            return y;
        if (xMagnitude > InfinityBits)
            return x;
        bool negativeX = (Interop.FloatAsInt(x) >> 63) != 0;
        if (yMagnitude == 0)
            return negativeX ? SignMagnitude(Pi, y) : y;
        double angle;
        if (yMagnitude == InfinityBits)
            angle = xMagnitude == InfinityBits ? negativeX ? 3 * QuarterPi : QuarterPi : HalfPi;
        else if (xMagnitude == InfinityBits)
            angle = negativeX ? Pi : 0;
        else
        {
            angle = Atan(Interop.IntAsFloat(yMagnitude) / Interop.IntAsFloat(xMagnitude));
            if (negativeX)
                angle = Pi - angle;
        }
        return SignMagnitude(angle, y);
    }

    private static double Asin(double value)
    {
        if ((Interop.FloatAsInt(value) & MagnitudeMask) > InfinityBits)
            return value;
        if (value < -1 || value > 1)
            return double.NaN;
        return Atan2(value, Math.Sqrt((1 - value) * (1 + value)));
    }

    private static double Acos(double value)
    {
        if ((Interop.FloatAsInt(value) & MagnitudeMask) > InfinityBits)
            return value;
        if (value < -1 || value > 1)
            return double.NaN;
        return Atan2(Math.Sqrt((1 - value) * (1 + value)), value);
    }

    /// <summary>Uses a small-argument series and shifts the large exponential before overflow can occur.</summary>
    private static double Sinh(double value)
    {
        ulong magnitude = Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude >= InfinityBits)
            return value;
        double positive = Interop.IntAsFloat(magnitude);
        if (positive < 1e-8)
            return value;
        double result;
        if (positive <= 1)
        {
            double square = positive * positive;
            double term = positive;
            result = positive;
            for (int n = 3; n <= 19; n += 2)
            {
                term *= square / ((n - 1) * n);
                result += term;
            }
        }
        else if (positive < 20)
            result = (Exp(positive) - Exp(-positive)) * 0.5;
        else
            result = Exp((positive - Ln2High) - Ln2Low);
        return SignMagnitude(result, value);
    }

    private static double Cosh(double value)
    {
        ulong magnitude = Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        if (magnitude == InfinityBits)
            return double.PositiveInfinity;
        double positive = Interop.IntAsFloat(magnitude);
        if (positive < 1e-8)
            return 1;
        if (positive <= 1)
        {
            double square = positive * positive;
            double term = 1;
            double result = 1;
            for (int n = 2; n <= 20; n += 2)
            {
                term *= square / ((n - 1) * n);
                result += term;
            }
            return result;
        }
        return positive < 20 ? (Exp(positive) + Exp(-positive)) * 0.5 : Exp((positive - Ln2High) - Ln2Low);
    }

    private static double Tanh(double value)
    {
        ulong magnitude = Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        double positive = Interop.IntAsFloat(magnitude);
        if (positive < 1e-8)
            return value;
        if (positive >= 20)
            return SignMagnitude(1, value);
        double result;
        if (positive <= 1)
            result = Sinh(positive) / Cosh(positive);
        else
        {
            double exponential = Exp(-2 * positive);
            result = (1 - exponential) / (1 + exponential);
        }
        return SignMagnitude(result, value);
    }

    /// <summary>Computes log(1+x) without discarding small x during addition to one.</summary>
    private static double LogOnePlus(double value)
    {
        if (Math.Abs(value) < 1e-16)
            return value;
        if (Math.Abs(value) > 0.5)
            return Log(1 + value);
        double z = value / (2 + value);
        double square = z * z;
        double series = 1.0 / 49;
        for (int denominator = 47; denominator >= 3; denominator -= 2)
            series = 1.0 / denominator + square * series;
        return 2 * z * (1 + square * series);
    }

    private static double Asinh(double value)
    {
        ulong magnitude = Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude >= InfinityBits)
            return value;
        double positive = Interop.IntAsFloat(magnitude);
        if (positive < 1e-8)
            return value;
        double result = positive > 1e154 ? Log(positive) + (Ln2High + Ln2Low) :
            LogOnePlus(positive + positive * positive / (1 + Math.Sqrt(1 + positive * positive)));
        return SignMagnitude(result, value);
    }

    private static double Acosh(double value)
    {
        if ((Interop.FloatAsInt(value) & MagnitudeMask) > InfinityBits)
            return value;
        if (value < 1)
            return double.NaN;
        if (value > 1e154)
            return Log(value) + (Ln2High + Ln2Low);
        double delta = value - 1;
        return LogOnePlus(delta + Math.Sqrt(delta * (value + 1)));
    }

    private static double Atanh(double value)
    {
        ulong magnitude = Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        double positive = Interop.IntAsFloat(magnitude);
        if (positive > 1)
            return double.NaN;
        if (positive == 1)
            return SignMagnitude(double.PositiveInfinity, value);
        if (positive < 1e-8)
            return value;
        return SignMagnitude(0.5 * LogOnePlus(2 * positive / (1 - positive)), value);
    }

    /// <summary>Starts with an exponential estimate and refines it without cubing an extreme input.</summary>
    private static double Cbrt(double value)
    {
        ulong magnitude = Interop.FloatAsInt(value) & MagnitudeMask;
        if (magnitude == 0 || magnitude >= InfinityBits)
            return value;
        double positive = Interop.IntAsFloat(magnitude);
        double root = Exp(Log(positive) / 3);
        for (int i = 0; i < 3; i++)
            root = (2 * root + (positive / root) / root) / 3;
        return SignMagnitude(root, value);
    }

    private static double LogBaseTwo(double value)
    {
        ulong bits = Interop.FloatAsInt(value);
        ulong magnitude = bits & MagnitudeMask;
        if (magnitude > InfinityBits)
            return value;
        if (magnitude == 0)
            return double.NegativeInfinity;
        if ((bits >> 63) != 0)
            return double.NaN;
        if (magnitude == InfinityBits)
            return value;
        return TryBinaryExponent(magnitude, out int exponent) ? exponent : Log(value) * InverseLn2;
    }
}
