using ILGPU;
using ILGPU.Runtime;
using System.Runtime.CompilerServices;

namespace ParallelGpu;

/// <summary>Provides atomic operations on shared device values in loop bodies and value kernels.</summary>
/// <remarks>Targets must refer to device memory shared by the participating iterations, such as an array
/// element. Ordinary captured scalars are read-only. Atomics order accesses to their target; they do not
/// publish a surrounding object graph or turn unrelated reads and writes into a transaction.
/// Floating-point updates follow thread arrival order and can vary between runs. Floating-point retry loops
/// compare bit patterns, so NaN targets do not prevent completion. Floating-point Min and Max follow the
/// backend's NaN and signed-zero semantics. CUDA addition retains its native atomic instruction.</remarks>
public static class GpuAtomic
{
    /// <summary>Selects addition in the floating-point compare-exchange fallback.</summary>
    private const int AddOperation = 0;
    /// <summary>Selects the backend's minimum operation in the floating-point fallback.</summary>
    private const int MinOperation = 1;
    /// <summary>Selects the backend's maximum operation in the floating-point fallback.</summary>
    private const int MaxOperation = 2;

    /// <summary>Atomically adds a value to the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Add(ref int target, int value) => Atomic.Add(ref target, value);

    /// <summary>Atomically replaces the target with the smaller value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Min(ref int target, int value) => Atomic.Min(ref target, value);

    /// <summary>Atomically replaces the target with the larger value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Max(ref int target, int value) => Atomic.Max(ref target, value);

    /// <summary>Atomically replaces the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Exchange(ref int target, int value) => Atomic.Exchange(ref target, value);

    /// <summary>Atomically replaces the target when it equals the comparison value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The replacement value.</param>
    /// <param name="comparand">The expected target value.</param>
    /// <returns>The previous target value, whether or not replacement succeeds.</returns>
    /// <remarks>The argument order matches System.Threading.Interlocked.CompareExchange.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int CompareExchange(ref int target, int value, int comparand) => Atomic.CompareExchange(ref target, comparand, value);

    /// <summary>Atomically adds a value to the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Add(ref long target, long value) => Atomic.Add(ref target, value);

    /// <summary>Atomically replaces the target with the smaller value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Min(ref long target, long value) => Atomic.Min(ref target, value);

    /// <summary>Atomically replaces the target with the larger value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Max(ref long target, long value) => Atomic.Max(ref target, value);

    /// <summary>Atomically replaces the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Exchange(ref long target, long value) => Atomic.Exchange(ref target, value);

    /// <summary>Atomically replaces the target when it equals the comparison value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The replacement value.</param>
    /// <param name="comparand">The expected target value.</param>
    /// <returns>The previous target value, whether or not replacement succeeds.</returns>
    /// <remarks>The argument order matches System.Threading.Interlocked.CompareExchange.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long CompareExchange(ref long target, long value, long comparand) => Atomic.CompareExchange(ref target, comparand, value);

    /// <summary>Atomically adds a value to the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Add(ref uint target, uint value) => Atomic.Add(ref target, value);

    /// <summary>Atomically replaces the target with the smaller value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Min(ref uint target, uint value) => Atomic.Min(ref target, value);

    /// <summary>Atomically replaces the target with the larger value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Max(ref uint target, uint value) => Atomic.Max(ref target, value);

    /// <summary>Atomically replaces the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Exchange(ref uint target, uint value) => Atomic.Exchange(ref target, value);

    /// <summary>Atomically replaces the target when it equals the comparison value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The replacement value.</param>
    /// <param name="comparand">The expected target value.</param>
    /// <returns>The previous target value, whether or not replacement succeeds.</returns>
    /// <remarks>The argument order matches System.Threading.Interlocked.CompareExchange.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint CompareExchange(ref uint target, uint value, uint comparand) => Atomic.CompareExchange(ref target, comparand, value);

    /// <summary>Atomically adds a value to the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Add(ref ulong target, ulong value) => Atomic.Add(ref target, value);

    /// <summary>Atomically replaces the target with the smaller value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Min(ref ulong target, ulong value) => Atomic.Min(ref target, value);

    /// <summary>Atomically replaces the target with the larger value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Max(ref ulong target, ulong value) => Atomic.Max(ref target, value);

    /// <summary>Atomically replaces the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Exchange(ref ulong target, ulong value) => Atomic.Exchange(ref target, value);

    /// <summary>Atomically replaces the target when it equals the comparison value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The replacement value.</param>
    /// <param name="comparand">The expected target value.</param>
    /// <returns>The previous target value, whether or not replacement succeeds.</returns>
    /// <remarks>The argument order matches System.Threading.Interlocked.CompareExchange.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong CompareExchange(ref ulong target, ulong value, ulong comparand) => Atomic.CompareExchange(ref target, comparand, value);

    /// <summary>Atomically adds a value to the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Add(ref float target, float value) => Accelerator.CurrentType == AcceleratorType.Cuda
        ? Atomic.Add(ref target, value) : UpdateFloating(ref target, value, AddOperation);

    /// <summary>Atomically replaces the target with the smaller value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(ref float target, float value) => UpdateFloating(ref target, value, MinOperation);

    /// <summary>Atomically replaces the target with the larger value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(ref float target, float value) => UpdateFloating(ref target, value, MaxOperation);

    /// <summary>Atomically replaces the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exchange(ref float target, float value) => Atomic.Exchange(ref target, value);

    /// <summary>Atomically replaces the target when its bits match the comparison value's bits.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The replacement value.</param>
    /// <param name="comparand">The expected target value.</param>
    /// <returns>The previous target value, whether or not replacement succeeds.</returns>
    /// <remarks>The argument order matches System.Threading.Interlocked.CompareExchange.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float CompareExchange(ref float target, float value, float comparand) => Atomic.CompareExchange(ref target, comparand, value);

    /// <summary>Atomically adds a value to the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Add(ref double target, double value) => Accelerator.CurrentType == AcceleratorType.Cuda
        ? Atomic.Add(ref target, value) : UpdateFloating(ref target, value, AddOperation);

    /// <summary>Atomically replaces the target with the smaller value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Min(ref double target, double value) => UpdateFloating(ref target, value, MinOperation);

    /// <summary>Atomically replaces the target with the larger value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Max(ref double target, double value) => UpdateFloating(ref target, value, MaxOperation);

    /// <summary>Updates a float with bitwise retry checks, avoiding ILGPU 1.5.3's NaN comparison failure.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The operand used by the update.</param>
    /// <param name="operation">The addition, minimum, or maximum operation selector.</param>
    /// <returns>The value held at the location immediately before the successful update.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float UpdateFloating(ref float target, float value, int operation)
    {
        float current = Atomic.CompareExchange(ref target, 0f, 0f);
        while (true)
        {
            float updated = operation == AddOperation ? current + value : operation == MinOperation
                ? IntrinsicMath.Min(current, value) : IntrinsicMath.Max(current, value);
            float previous = Atomic.CompareExchange(ref target, current, updated);
            // Floating equality cannot identify a successful NaN CAS and also conflates the two signed zeros.
            if (Interop.FloatAsInt(previous) == Interop.FloatAsInt(current))
                return previous;
            current = previous;
        }
    }

    /// <summary>Updates a double with the same NaN-safe, bitwise retry rule as the float overload.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The operand used by the update.</param>
    /// <param name="operation">The addition, minimum, or maximum operation selector.</param>
    /// <returns>The value held at the location immediately before the successful update.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double UpdateFloating(ref double target, double value, int operation)
    {
        double current = Atomic.CompareExchange(ref target, 0d, 0d);
        while (true)
        {
            double updated = operation == AddOperation ? current + value : operation == MinOperation
                ? IntrinsicMath.Min(current, value) : IntrinsicMath.Max(current, value);
            double previous = Atomic.CompareExchange(ref target, current, updated);
            if (Interop.FloatAsInt(previous) == Interop.FloatAsInt(current))
                return previous;
            current = previous;
        }
    }

    /// <summary>Atomically replaces the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Exchange(ref double target, double value) => Atomic.Exchange(ref target, value);

    /// <summary>Atomically replaces the target when its bits match the comparison value's bits.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The replacement value.</param>
    /// <param name="comparand">The expected target value.</param>
    /// <returns>The previous target value, whether or not replacement succeeds.</returns>
    /// <remarks>The argument order matches System.Threading.Interlocked.CompareExchange.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double CompareExchange(ref double target, double value, double comparand) => Atomic.CompareExchange(ref target, comparand, value);

    /// <summary>Atomically applies bitwise AND and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int And(ref int target, int value) => Atomic.And(ref target, value);

    /// <summary>Atomically applies bitwise OR and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Or(ref int target, int value) => Atomic.Or(ref target, value);

    /// <summary>Atomically applies bitwise XOR and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Xor(ref int target, int value) => Atomic.Xor(ref target, value);

    /// <summary>Atomically applies bitwise AND and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long And(ref long target, long value) => Atomic.And(ref target, value);

    /// <summary>Atomically applies bitwise OR and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Or(ref long target, long value) => Atomic.Or(ref target, value);

    /// <summary>Atomically applies bitwise XOR and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Xor(ref long target, long value) => Atomic.Xor(ref target, value);

    /// <summary>Atomically applies bitwise AND and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint And(ref uint target, uint value) => Atomic.And(ref target, value);

    /// <summary>Atomically applies bitwise OR and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Or(ref uint target, uint value) => Atomic.Or(ref target, value);

    /// <summary>Atomically applies bitwise XOR and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Xor(ref uint target, uint value) => Atomic.Xor(ref target, value);

    /// <summary>Atomically applies bitwise AND and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong And(ref ulong target, ulong value) => Atomic.And(ref target, value);

    /// <summary>Atomically applies bitwise OR and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Or(ref ulong target, ulong value) => Atomic.Or(ref target, value);

    /// <summary>Atomically applies bitwise XOR and returns the previous target value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The bit mask.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Xor(ref ulong target, ulong value) => Atomic.Xor(ref target, value);

}
