namespace ParallelGpu;

/// <summary>A value-type operation executed once per buffer element.</summary>
public interface IGpuForEachKernel<T> where T : unmanaged
{
    /// <summary>Changes to <paramref name="value"/> are written to the GPU buffer.
    /// The index is relative to the start of the buffer.</summary>
    void Execute(int index, ref T value);
}
