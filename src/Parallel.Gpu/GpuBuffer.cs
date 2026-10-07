using ILGPU;
using ILGPU.Runtime;

namespace ParallelGpu;

/// <summary>A disposable, device-resident buffer. Dispose it before its owner,
/// or let the owner release it. Stored views must not outlive the buffer.</summary>
public sealed class GpuBuffer<T> : IDisposable where T : unmanaged
{
    private readonly GpuParallel owner;
    private readonly MemoryBuffer1D<T, Stride1D.Dense> memory;
    private bool disposed;

    internal GpuBuffer(GpuParallel owner, MemoryBuffer1D<T, Stride1D.Dense> memory, int length)
    {
        this.owner = owner;
        this.memory = memory;
        Length = length;
    }

    public int Length { get; }

    /// <summary>Use this view only in kernels executed by this buffer's owner.</summary>
    public ArrayView<T> View
    {
        get
        {
            lock (owner.SyncRoot)
            {
                ThrowIfDisposed();
                return memory.View.BaseView;
            }
        }
    }

    public T[] ToArray()
    {
        var result = new T[Length];
        CopyTo(result);
        return result;
    }

    public void CopyFrom(T[] source)
    {
        if (source is null)
            throw new ArgumentNullException(nameof(source));
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (source.Length != Length)
                throw new ArgumentException("Source length must match the buffer.", nameof(source));
            if (Length > 0)
                memory.CopyFromCPU(source);
        }
    }

    public void CopyTo(T[] destination)
    {
        if (destination is null)
            throw new ArgumentNullException(nameof(destination));
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (destination.Length != Length)
                throw new ArgumentException("Destination length must match the buffer.", nameof(destination));
            if (Length > 0)
                memory.CopyToCPU(destination);
        }
    }

    internal void CopyTo(int sourceOffset, T[] destination)
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (sourceOffset < 0 || sourceOffset > Length || destination.Length > Length - sourceOffset)
                throw new ArgumentOutOfRangeException(nameof(sourceOffset));
            if (destination.Length > 0)
                memory.View.BaseView.SubView(sourceOffset, destination.Length).CopyToCPU(destination);
        }
    }

    public void Dispose()
    {
        lock (owner.SyncRoot)
        {
            if (disposed)
                return;
            memory.Dispose();
            disposed = true;
            owner.Release(this);
        }
    }

    internal void ValidateOwner(GpuParallel expectedOwner)
    {
        if (!ReferenceEquals(owner, expectedOwner))
            throw new ArgumentException("The buffer belongs to another GpuParallel instance.");
        ThrowIfDisposed();
    }

    private void ThrowIfDisposed()
    {
        if (disposed)
            throw new ObjectDisposedException(GetType().FullName);
        owner.ThrowIfDisposed();
    }
}
