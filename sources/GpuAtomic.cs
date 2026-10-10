using ILGPU;
using System.Runtime.CompilerServices;

namespace ParallelGpu;

/// <summary>Provides atomic operations on shared device values in loop bodies and value kernels.</summary>
/// <remarks>Targets must refer to device memory shared by the participating iterations, such as an array
/// element. Ordinary captured scalars are read-only. Atomics order accesses to their target; they do not
/// publish a surrounding object graph or turn unrelated reads and writes into a transaction.
/// Floating-point updates follow thread arrival order and can vary between runs.</remarks>
public static class GpuAtomic
{
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
    public static float Add(ref float target, float value) => Atomic.Add(ref target, value);

    /// <summary>Atomically replaces the target with the smaller value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(ref float target, float value) => Atomic.Min(ref target, value);

    /// <summary>Atomically replaces the target with the larger value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(ref float target, float value) => Atomic.Max(ref target, value);

    /// <summary>Atomically replaces the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exchange(ref float target, float value) => Atomic.Exchange(ref target, value);

    /// <summary>Atomically replaces the target when it equals the comparison value.</summary>
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
    public static double Add(ref double target, double value) => Atomic.Add(ref target, value);

    /// <summary>Atomically replaces the target with the smaller value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Min(ref double target, double value) => Atomic.Min(ref target, value);

    /// <summary>Atomically replaces the target with the larger value and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Max(ref double target, double value) => Atomic.Max(ref target, value);

    /// <summary>Atomically replaces the target and returns its previous value.</summary>
    /// <param name="target">The shared device location.</param>
    /// <param name="value">The value used by the operation.</param>
    /// <returns>The previous target value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Exchange(ref double target, double value) => Atomic.Exchange(ref target, value);

    /// <summary>Atomically replaces the target when it equals the comparison value.</summary>
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
