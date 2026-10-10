namespace ParallelGpu;

/// <summary>Defines an associative reduction with a two-sided identity value.</summary>
/// <typeparam name="T">The public unmanaged, ILGPU-supported value type.</typeparam>
/// <remarks>Implementations must be public value types with device-compatible state and pure methods.
/// Combine must be associative and Identity must be neutral on both sides. Input order is preserved,
/// so commutativity is not required. Floating-point operations follow the reduction tree's rounding.</remarks>
public interface IGpuReduction<T> where T : unmanaged
{
    /// <summary>Gets the neutral value returned when the input is empty.</summary>
    T Identity { get; }

    /// <summary>Combines two consecutive portions of the input.</summary>
    /// <param name="left">The reduction of the earlier portion.</param>
    /// <param name="right">The reduction of the later portion.</param>
    /// <returns>The reduction of their concatenation.</returns>
    T Combine(T left, T right);
}

/// <summary>Sums signed 32-bit integers with unchecked overflow.</summary>
public readonly struct Int32Sum : IGpuReduction<int>
{
    /// <inheritdoc/>
    public int Identity => 0;
    /// <inheritdoc/>
    public int Combine(int left, int right) => unchecked(left + right);
}

/// <summary>Sums signed 64-bit integers with unchecked overflow.</summary>
public readonly struct Int64Sum : IGpuReduction<long>
{
    /// <inheritdoc/>
    public long Identity => 0;
    /// <inheritdoc/>
    public long Combine(long left, long right) => unchecked(left + right);
}

/// <summary>Sums unsigned 32-bit integers with unchecked overflow.</summary>
public readonly struct UInt32Sum : IGpuReduction<uint>
{
    /// <inheritdoc/>
    public uint Identity => 0;
    /// <inheritdoc/>
    public uint Combine(uint left, uint right) => unchecked(left + right);
}

/// <summary>Sums unsigned 64-bit integers with unchecked overflow.</summary>
public readonly struct UInt64Sum : IGpuReduction<ulong>
{
    /// <inheritdoc/>
    public ulong Identity => 0;
    /// <inheritdoc/>
    public ulong Combine(ulong left, ulong right) => unchecked(left + right);
}

/// <summary>Sums 32-bit floating-point values in reduction-tree order.</summary>
public readonly struct Float32Sum : IGpuReduction<float>
{
    /// <inheritdoc/>
    public float Identity => 0;
    /// <inheritdoc/>
    public float Combine(float left, float right) => left + right;
}

/// <summary>Sums 64-bit floating-point values in reduction-tree order.</summary>
public readonly struct Float64Sum : IGpuReduction<double>
{
    /// <inheritdoc/>
    public double Identity => 0;
    /// <inheritdoc/>
    public double Combine(double left, double right) => left + right;
}
