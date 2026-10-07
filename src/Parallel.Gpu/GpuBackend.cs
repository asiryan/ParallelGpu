namespace ParallelGpu;

/// <summary>Specifies the accelerator backend used to execute parallel loops.</summary>
public enum GpuBackend
{
    /// <summary>Selects CUDA first, then OpenCL, with optional CPU fallback.</summary>
    /// <remarks>CPU fallback requires <see cref="GpuOptions.AllowCpuFallback"/>.</remarks>
    Auto,
    /// <summary>Uses an NVIDIA CUDA accelerator.</summary>
    Cuda,
    /// <summary>Uses a GPU accelerator provided by an OpenCL runtime.</summary>
    OpenCL,
    /// <summary>Uses the ILGPU CPU accelerator for debugging and correctness checks.</summary>
    Cpu
}
