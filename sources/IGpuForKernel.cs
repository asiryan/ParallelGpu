namespace ParallelGpu;

/// <summary>Defines a device-compatible operation executed once for each loop index.</summary>
/// <remarks>Implement this interface in a public value type containing GPU views and supported values.
/// Managed objects and host API calls cannot be used inside the operation. State uses its native ILGPU
/// representation; ordinary lambda value conversion and deferred-error handling do not apply.</remarks>
public interface IGpuForKernel
{
    /// <summary>Executes the operation for one index in the requested range.</summary>
    /// <param name="index">The original loop index, including the range's lower bound.</param>
    void Execute(int index);
}
