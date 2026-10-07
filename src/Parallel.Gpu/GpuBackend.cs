namespace ParallelGpu;

public enum GpuBackend
{
    /// <summary>Try CUDA, then OpenCL. CPU fallback requires explicit permission.</summary>
    Auto,
    Cuda,
    OpenCL,
    /// <summary>ILGPU's CPU accelerator, useful for correctness tests and debugging.</summary>
    Cpu
}
