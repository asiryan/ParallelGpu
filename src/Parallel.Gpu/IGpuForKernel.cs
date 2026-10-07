namespace ParallelGpu;

/// <summary>A value-type operation executed once for every index in a range.</summary>
/// <remarks>Fields may contain GPU buffer views and ILGPU-supported value types.
/// Managed objects, captured delegates and host API calls are not supported.</remarks>
public interface IGpuForKernel
{
    void Execute(int index);
}
