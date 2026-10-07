namespace ParallelGpu;

/// <summary>Defines a device-compatible operation that processes buffer elements by reference.</summary>
/// <typeparam name="T">The unmanaged, ILGPU-supported element type.</typeparam>
/// <remarks>Implement this interface in a public value type containing GPU views and supported values.</remarks>
public interface IGpuForEachKernel<T> where T : unmanaged
{
    /// <summary>Processes one element and writes changes through its reference.</summary>
    /// <param name="index">The zero-based element index relative to the beginning of the buffer.</param>
    /// <param name="value">A reference to the device-resident element.</param>
    void Execute(int index, ref T value);
}
