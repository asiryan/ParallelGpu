namespace ParallelGpu;

/// <summary>Configures accelerator selection and optional CPU fallback.</summary>
public sealed record GpuOptions
{
    /// <summary>Gets or initializes the requested backend. The default is <see cref="GpuBackend.Auto"/>.</summary>
    public GpuBackend Backend { get; init; } = GpuBackend.Auto;

    /// <summary>Gets or initializes the zero-based device index within the selected backend.</summary>
    /// <remarks>In Auto mode, zero tries all GPUs in CUDA-first order; a larger index selects one GPU.</remarks>
    public int DeviceIndex { get; init; }

    /// <summary>Gets or initializes whether Auto mode may use the CPU accelerator when no GPU can be initialized.</summary>
    /// <remarks>The default is false. Explicit CUDA and OpenCL requests never fall back to the CPU.</remarks>
    public bool AllowCpuFallback { get; init; }
}
