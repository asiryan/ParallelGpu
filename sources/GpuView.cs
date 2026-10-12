using ILGPU;
using System.Runtime.CompilerServices;

namespace ParallelGpu;

/// <summary>Provides indexed device access to a buffer without exposing its backend representation.</summary>
/// <typeparam name="T">The unmanaged device element type.</typeparam>
/// <remarks>The view borrows memory from a <see cref="GpuBuffer{T}"/> and cannot outlive it or its executor.
/// Element access is intended for GPU loop bodies and explicit value kernels; use the buffer's copy methods
/// for host access. Lambda captures validate the owner and lifetime before execution and require a public
/// payload that needs no automatic value conversion. In both lambdas and explicit kernels, indices must be
/// within Length; element access follows native device contracts without deferred bounds checks.
/// Device buffer writes are not rolled back on a later error. Explicit kernels also require the caller
/// to keep the view alive and use it only with its owning executor.</remarks>
public readonly struct GpuView<T> : IDeviceNativeView where T : unmanaged
{
    /// <summary>The allocation lifetime checked before an automatic lambda launch.</summary>
    private readonly int lifetimeId;
    /// <summary>The private backend view over the borrowed device allocation.</summary>
    private readonly ArrayView<T> view;

    /// <summary>Wraps device memory whose owner has already validated its allocation.</summary>
    /// <param name="lifetimeId">The identity of the live buffer lease.</param>
    /// <param name="view">The backend device memory descriptor.</param>
    internal GpuView(int lifetimeId, ArrayView<T> view)
    {
        this.lifetimeId = lifetimeId;
        this.view = view;
    }

    /// <summary>Gets the number of elements accessible through this view.</summary>
    public int Length => view.IntLength;

    /// <summary>Gets a device element by reference for reads, writes, and atomic updates inside a kernel.</summary>
    /// <param name="index">The zero-based index, less than <see cref="Length"/>.</param>
    /// <returns>A reference to the device-resident element.</returns>
    public ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref view[index];
    }

    /// <summary>Validates ownership and the borrowed allocation's lifetime before a lambda launch.</summary>
    /// <param name="owner">The executor about to use this captured view.</param>
    void IDeviceNativeView.ValidateOwner(Gpu owner) => GpuViewLifetime.Validate(lifetimeId, owner);
}
