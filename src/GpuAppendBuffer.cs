using ILGPU;

namespace ParallelGpu;

/// <summary>A device view that appends values into a host-allocated bounded buffer.</summary>
/// <typeparam name="T">The unmanaged, ILGPU-supported element type.</typeparam>
/// <remarks>Concurrent appends reserve distinct slots atomically. Their order is unspecified.
/// Read the output on the host only after the kernel has completed. Lambda captures validate ownership
/// and lifetime before launching. Explicit value kernels must use views with their owning executor
/// and within their owner's lifetime.</remarks>
public readonly struct GpuAppendView<T> : IDeviceNativeView where T : unmanaged
{
    /// <summary>The host-validated allocation lifetime.</summary>
    private readonly int lifetimeId;
    /// <summary>The host-allocated device slots.</summary>
    private readonly ArrayView<T> values;
    /// <summary>The saturated count followed by the overflow flag.</summary>
    private readonly ArrayView<int> state;

    /// <summary>Creates a view over its owner's validated allocations.</summary>
    /// <param name="lifetimeId">The allocation lifetime identity.</param>
    /// <param name="values">The available slots.</param>
    /// <param name="state">The count and overflow flag.</param>
    internal GpuAppendView(int lifetimeId, ArrayView<T> values, ArrayView<int> state)
    {
        this.lifetimeId = lifetimeId;
        this.values = values;
        this.state = state;
    }

    /// <summary>Validates this capture before the host launches a lambda kernel.</summary>
    /// <param name="owner">The launching executor.</param>
    void IDeviceNativeView.ValidateOwner(Gpu owner) => GpuViewLifetime.Validate(lifetimeId, owner);

    /// <summary>Gets the maximum number of output values.</summary>
    public int Capacity => values.IntLength;

    /// <summary>Appends a value, recording overflow if no slot remains.</summary>
    /// <param name="value">The value to append.</param>
    /// <remarks>Overflow never writes outside the buffer. The host owner's ToArray method rejects
    /// an overflowed result. Use TryAdd when the kernel needs to branch on capacity exhaustion.</remarks>
    public void Add(T value) => TryAdd(value);

    /// <summary>Attempts to append a value and records overflow on capacity exhaustion.</summary>
    /// <param name="value">The value to append.</param>
    /// <returns>True if the value was written into a distinct slot; otherwise false.</returns>
    public bool TryAdd(T value) => TryAdd(value, out _);

    /// <summary>Attempts to append a value and reports its reserved slot.</summary>
    /// <param name="value">The value to append.</param>
    /// <param name="index">The zero-based reserved slot, or -1 when capacity is exhausted.</param>
    /// <returns>True if the value was appended; otherwise false.</returns>
    /// <remarks>Reservation does not publish the value. Other device threads need a synchronization
    /// boundary before reading the appended value.</remarks>
    public bool TryAdd(T value, out int index)
    {
        int capacity = values.IntLength;
        int current = Atomic.Add(ref state[0], 0);
        while (current < capacity)
        {
            int observed = Atomic.CompareExchange(ref state[0], current, current + 1);
            if (observed == current)
            {
                values[current] = value;
                index = current;
                return true;
            }
            current = observed;
        }
        Atomic.Exchange(ref state[1], 1);
        index = -1;
        return false;
    }
}

/// <summary>Owns a reusable bounded output buffer for concurrent device append operations.</summary>
/// <typeparam name="T">The unmanaged, ILGPU-supported element type.</typeparam>
/// <remarks>Capacity is fixed on the host. Kernels append through View. ToArray rejects overflowed
/// output; Reset permits reuse. Its Gpu owner also disposes the underlying allocations.</remarks>
public sealed class GpuAppendBuffer<T> : IDisposable where T : unmanaged
{
    /// <summary>The executor that serializes host access.</summary>
    private readonly Gpu owner;
    /// <summary>The lifetime shared by all copied device views.</summary>
    private readonly GpuViewLifetime lifetime;
    /// <summary>The device output slots.</summary>
    private readonly GpuBuffer<T> values;
    /// <summary>The device count and overflow flag.</summary>
    private readonly GpuBuffer<int> state;
    /// <summary>Whether this wrapper has been disposed.</summary>
    private bool disposed;

    /// <summary>Wraps two allocations belonging to one executor.</summary>
    /// <param name="owner">The owning executor.</param>
    /// <param name="values">The output slots.</param>
    /// <param name="state">The count and overflow flag.</param>
    internal GpuAppendBuffer(Gpu owner, GpuBuffer<T> values, GpuBuffer<int> state)
    {
        this.owner = owner;
        lifetime = new(owner);
        this.values = values;
        this.state = state;
    }

    /// <summary>Gets the maximum output length.</summary>
    public int Capacity => values.Length;

    /// <summary>Gets a device view for kernels executed by the owning executor.</summary>
    /// <exception cref="ObjectDisposedException">The buffer or executor has been disposed.</exception>
    public GpuAppendView<T> View
    {
        get
        {
            lock (owner.SyncRoot)
            {
                ThrowIfDisposed();
                return new(lifetime.Id, values.View, state.View);
            }
        }
    }

    /// <summary>Gets the number of successfully reserved output slots.</summary>
    /// <remarks>The count saturates at Capacity. Query it only after a synchronous kernel call.</remarks>
    /// <exception cref="ObjectDisposedException">The buffer or executor has been disposed.</exception>
    public int Count
    {
        get
        {
            lock (owner.SyncRoot)
            {
                ThrowIfDisposed();
                return state.ToArray()[0];
            }
        }
    }

    /// <summary>Gets whether any append exceeded capacity since the last reset.</summary>
    /// <exception cref="ObjectDisposedException">The buffer or executor has been disposed.</exception>
    public bool Overflowed
    {
        get
        {
            lock (owner.SyncRoot)
            {
                ThrowIfDisposed();
                return state.ToArray()[1] != 0;
            }
        }
    }

    /// <summary>Copies successfully appended values to a new host array.</summary>
    /// <returns>The output values in their reservation order.</returns>
    /// <exception cref="InvalidOperationException">At least one append exceeded capacity.</exception>
    /// <exception cref="ObjectDisposedException">The buffer or executor has been disposed.</exception>
    public T[] ToArray()
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            var status = state.ToArray();
            if (status[1] != 0)
                throw new InvalidOperationException($"GPU output exceeded the bounded capacity of {Capacity} elements. Reset the buffer before reuse.");
            var result = new T[status[0]];
            values.CopyTo(0, result);
            return result;
        }
    }

    /// <summary>Clears the output count and overflow flag so the slots can be reused.</summary>
    /// <remarks>Previous slot contents are inaccessible through ToArray after reset.</remarks>
    /// <exception cref="ObjectDisposedException">The buffer or executor has been disposed.</exception>
    public void Reset()
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            state.CopyFrom(new int[2]);
        }
    }

    /// <summary>Releases the output slots and state; repeated calls have no effect.</summary>
    public void Dispose()
    {
        lock (owner.SyncRoot)
        {
            if (disposed)
                return;
            lifetime.Dispose();
            state.Dispose();
            values.Dispose();
            disposed = true;
            owner.Release(this);
        }
    }

    /// <summary>Checks the owner and wrapper lifetime before host access.</summary>
    private void ThrowIfDisposed()
    {
        if (disposed)
            throw new ObjectDisposedException(GetType().FullName);
        owner.ThrowIfDisposed();
    }
}

/// <summary>Produces a variable number of output values per indexed iteration.</summary>
/// <typeparam name="T">The unmanaged, ILGPU-supported output type.</typeparam>
/// <remarks>Implementations must be public value types with device-compatible state.
/// Values can be newly constructed structures; storage is reserved from a fixed-capacity host allocation.</remarks>
public interface IGpuCollectKernel<T> where T : unmanaged
{
    /// <summary>Processes an index and appends zero or more output values.</summary>
    /// <param name="index">The original loop index.</param>
    /// <param name="output">The concurrent bounded output view.</param>
    void Execute(int index, GpuAppendView<T> output);
}
