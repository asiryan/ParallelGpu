using System.Collections.Concurrent;

namespace ParallelGpu;

/// <summary>Validates an explicitly owned native view before a lambda kernel is launched.</summary>
internal interface IDeviceNativeView
{
    /// <summary>Rejects default, expired, and foreign-executor views on the host.</summary>
    /// <param name="owner">The executor about to launch the kernel.</param>
    void ValidateOwner(Gpu owner);
}

/// <summary>Tracks the lifetime of host-owned native views without retaining abandoned executors.</summary>
internal sealed class GpuViewLifetime : IDisposable
{
    /// <summary>Weak leases indexed by the identity copied into device views.</summary>
    private static readonly ConcurrentDictionary<int, WeakReference<GpuViewLifetime>> leases = new();
    /// <summary>The owning executor.</summary>
    private readonly Gpu owner;
    /// <summary>Whether the view's allocations have been revoked.</summary>
    private bool disposed;

    /// <summary>Registers a lifetime owned by a native view wrapper.</summary>
    /// <param name="owner">The allocation owner.</param>
    internal GpuViewLifetime(Gpu owner)
    {
        this.owner = owner;
        Id = GpuDeviceIdentity.Allocate();
        leases.TryAdd(Id, new(this));
    }

    /// <summary>Removes an abandoned weak registration when its executor is collected.</summary>
    ~GpuViewLifetime() => leases.TryRemove(Id, out _);

    /// <summary>Gets the unique identity passed to device views.</summary>
    internal int Id { get; }

    /// <summary>Validates a copied native view against its allocation owner's current lifetime.</summary>
    /// <param name="id">The copied lifetime identity.</param>
    /// <param name="expectedOwner">The executor about to launch the kernel.</param>
    internal static void Validate(int id, Gpu expectedOwner)
    {
        if (!leases.TryGetValue(id, out var weak) || !weak.TryGetTarget(out var lease))
            throw new ObjectDisposedException("GPU device view", "The native view is default or its allocations have been disposed.");
        if (!ReferenceEquals(lease.owner, expectedOwner))
            throw new ArgumentException("The native device view belongs to another Gpu instance.");
        lock (expectedOwner.SyncRoot)
        {
            expectedOwner.ThrowIfDisposed();
            if (lease.disposed)
                throw new ObjectDisposedException("GPU device view");
        }
    }

    /// <summary>Revokes existing view copies before their allocations are released.</summary>
    public void Dispose()
    {
        lock (owner.SyncRoot)
        {
            if (disposed)
                return;
            disposed = true;
            leases.TryRemove(Id, out _);
            GC.SuppressFinalize(this);
        }
    }
}
