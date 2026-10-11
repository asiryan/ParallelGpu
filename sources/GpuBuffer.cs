using ILGPU;
using ILGPU.Runtime;

namespace ParallelGpu;

/// <summary>Owns a device-resident buffer allocated by a <see cref="Gpu"/> instance.</summary>
/// <typeparam name="T">The unmanaged, ILGPU-supported element type.</typeparam>
/// <remarks>Host copies wait for completion and use the native element representation without automatic
/// value conversion. Dispose the buffer when it is no longer needed; its owner also releases outstanding
/// buffers. Stored views must not be used after the buffer or its owner has been disposed.</remarks>
public sealed class GpuBuffer<T> : IDisposable where T : unmanaged
{
    /// <summary>The executor that owns this allocation and serializes access to it.</summary>
    private readonly Gpu owner;
    /// <summary>The underlying ILGPU allocation released by this buffer.</summary>
    private readonly MemoryBuffer1D<T, Stride1D.Dense> memory;
    /// <summary>Whether the device allocation has been released.</summary>
    private bool disposed;

    /// <summary>Wraps an allocated device buffer and associates it with its owner.</summary>
    /// <param name="owner">The executor that owns the allocation.</param>
    /// <param name="memory">The device memory allocation.</param>
    /// <param name="length">The number of elements in the allocation.</param>
    internal GpuBuffer(Gpu owner, MemoryBuffer1D<T, Stride1D.Dense> memory, int length)
    {
        this.owner = owner;
        this.memory = memory;
        Length = length;
    }

    /// <summary>Gets the number of elements in the buffer.</summary>
    public int Length { get; }

    /// <summary>Gets the device view used by kernels executed by this buffer's owner.</summary>
    /// <remarks>The view does not own memory and must not outlive the buffer.</remarks>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
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

    /// <summary>Copies the entire buffer into a newly allocated host array.</summary>
    /// <returns>An array containing the current buffer contents.</returns>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
    public T[] ToArray()
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            var result = new T[Length];
            CopyTo(result);
            return result;
        }
    }

    /// <summary>Replaces the buffer contents with data from a host array.</summary>
    /// <param name="source">An array whose length equals <see cref="Length"/>.</param>
    /// <exception cref="ArgumentNullException">The source array is null.</exception>
    /// <exception cref="ArgumentException">The source length differs from the buffer length.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
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

    /// <summary>Copies a host array into a contiguous range of the device buffer.</summary>
    /// <param name="destinationOffset">The zero-based offset of the first destination element.</param>
    /// <param name="source">The host array containing the elements to copy.</param>
    /// <remarks>The call waits for the copy to complete. An empty source is allowed at the end of the buffer.</remarks>
    /// <exception cref="ArgumentNullException">The source array is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The requested range lies outside the buffer.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
    public void CopyFrom(int destinationOffset, T[] source)
    {
        if (source is null)
            throw new ArgumentNullException(nameof(source));
        CopyFrom(destinationOffset, (ReadOnlySpan<T>)source);
    }

    /// <summary>Copies a host array range into a contiguous range of the device buffer.</summary>
    /// <param name="destinationOffset">The zero-based offset of the first destination element in the buffer.</param>
    /// <param name="source">The host array containing the elements to copy.</param>
    /// <param name="sourceOffset">The zero-based offset of the first source element in the array.</param>
    /// <param name="count">The number of elements to copy.</param>
    /// <remarks>The call waits for the copy to complete. A zero count is allowed at the end of either range.</remarks>
    /// <exception cref="ArgumentNullException">The source array is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An offset or count lies outside its source or destination.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
    public void CopyFrom(int destinationOffset, T[] source, int sourceOffset, int count)
    {
        if (source is null)
            throw new ArgumentNullException(nameof(source));
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (destinationOffset < 0 || destinationOffset > Length)
                throw new ArgumentOutOfRangeException(nameof(destinationOffset));
            if (sourceOffset < 0 || sourceOffset > source.Length)
                throw new ArgumentOutOfRangeException(nameof(sourceOffset));
            if (count < 0 || count > Length - destinationOffset || count > source.Length - sourceOffset)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count > 0)
            {
                ReadOnlySpan<T> range = source.AsSpan(sourceOffset, count);
                memory.View.BaseView.SubView(destinationOffset, count).CopyFromCPU(in range);
            }
        }
    }

    /// <summary>Copies a host span into a contiguous range of the device buffer.</summary>
    /// <param name="destinationOffset">The zero-based offset of the first destination element.</param>
    /// <param name="source">The host span containing the elements to copy.</param>
    /// <remarks>The call waits for the copy to complete and does not retain the span.
    /// An empty source is allowed at the end of the buffer.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">The requested range lies outside the buffer.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
    public void CopyFrom(int destinationOffset, ReadOnlySpan<T> source)
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (destinationOffset < 0 || destinationOffset > Length || source.Length > Length - destinationOffset)
                throw new ArgumentOutOfRangeException(nameof(destinationOffset));
            if (source.Length > 0)
                memory.View.BaseView.SubView(destinationOffset, source.Length).CopyFromCPU(in source);
        }
    }

    /// <summary>Copies the entire buffer into an existing host array.</summary>
    /// <param name="destination">An array whose length equals <see cref="Length"/>.</param>
    /// <exception cref="ArgumentNullException">The destination array is null.</exception>
    /// <exception cref="ArgumentException">The destination length differs from the buffer length.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
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

    /// <summary>Copies a contiguous buffer range into a host array.</summary>
    /// <param name="sourceOffset">The zero-based offset of the first source element.</param>
    /// <param name="destination">The array that determines the number of elements to copy.</param>
    /// <remarks>The call waits for the copy to complete. An empty destination is allowed at the end of the buffer.</remarks>
    /// <exception cref="ArgumentNullException">The destination array is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The requested range lies outside the buffer.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
    public void CopyTo(int sourceOffset, T[] destination)
    {
        if (destination is null)
            throw new ArgumentNullException(nameof(destination));
        CopyTo(sourceOffset, (Span<T>)destination);
    }

    /// <summary>Copies a contiguous buffer range into a range of a host array.</summary>
    /// <param name="sourceOffset">The zero-based offset of the first source element in the buffer.</param>
    /// <param name="destination">The host array receiving the elements.</param>
    /// <param name="destinationOffset">The zero-based offset of the first destination element in the array.</param>
    /// <param name="count">The number of elements to copy.</param>
    /// <remarks>The call waits for the copy to complete. A zero count is allowed at the end of either range.</remarks>
    /// <exception cref="ArgumentNullException">The destination array is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An offset or count lies outside its source or destination.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
    public void CopyTo(int sourceOffset, T[] destination, int destinationOffset, int count)
    {
        if (destination is null)
            throw new ArgumentNullException(nameof(destination));
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (sourceOffset < 0 || sourceOffset > Length)
                throw new ArgumentOutOfRangeException(nameof(sourceOffset));
            if (destinationOffset < 0 || destinationOffset > destination.Length)
                throw new ArgumentOutOfRangeException(nameof(destinationOffset));
            if (count < 0 || count > Length - sourceOffset || count > destination.Length - destinationOffset)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count > 0)
            {
                Span<T> range = destination.AsSpan(destinationOffset, count);
                memory.View.BaseView.SubView(sourceOffset, count).CopyToCPU(in range);
            }
        }
    }

    /// <summary>Copies a contiguous buffer range into a host span.</summary>
    /// <param name="sourceOffset">The zero-based offset of the first source element.</param>
    /// <param name="destination">The span that determines the number of elements to copy.</param>
    /// <remarks>The call waits for the copy to complete and does not retain the span.
    /// An empty destination is allowed at the end of the buffer.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">The requested range lies outside the buffer.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
    public void CopyTo(int sourceOffset, Span<T> destination)
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (sourceOffset < 0 || sourceOffset > Length || destination.Length > Length - sourceOffset)
                throw new ArgumentOutOfRangeException(nameof(sourceOffset));
            if (destination.Length > 0)
                memory.View.BaseView.SubView(sourceOffset, destination.Length).CopyToCPU(in destination);
        }
    }

    /// <summary>Releases the device allocation and unregisters the buffer from its owner.</summary>
    /// <remarks>Repeated calls have no effect.</remarks>
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

    /// <summary>Checks that the buffer is alive and belongs to the requested executor.</summary>
    /// <param name="expectedOwner">The executor that will use the buffer.</param>
    /// <exception cref="ArgumentException">The buffer belongs to another executor.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or its owner has been disposed.</exception>
    internal void ValidateOwner(Gpu expectedOwner)
    {
        if (!ReferenceEquals(owner, expectedOwner))
            throw new ArgumentException("The buffer belongs to another Gpu instance.");
        ThrowIfDisposed();
    }

    /// <summary>Rejects access after the buffer or its owner has been disposed.</summary>
    private void ThrowIfDisposed()
    {
        if (disposed)
            throw new ObjectDisposedException(GetType().FullName);
        owner.ThrowIfDisposed();
    }
}
