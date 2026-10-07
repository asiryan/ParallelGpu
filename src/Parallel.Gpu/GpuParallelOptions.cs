namespace ParallelGpu;

public sealed record GpuParallelOptions
{
    public GpuBackend Backend { get; init; } = GpuBackend.Auto;

    /// <summary>Zero-based device index within the requested backend.
    /// In Auto mode it indexes the CUDA-first list of GPU devices.</summary>
    public int DeviceIndex { get; init; }

    /// <summary>Allow Auto mode to use ILGPU's CPU accelerator if no GPU can start.
    /// Explicit Cuda/OpenCL requests never silently fall back.</summary>
    public bool AllowCpuFallback { get; init; }
}
