using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace ParallelGpu;

/// <summary>Owns an ILGPU accelerator and executes synchronous parallel loops.</summary>
/// <remarks>Calls on one instance are serialized. Loop methods wait for device execution and required
/// host copies to complete. Reuse one instance to retain compiled kernels between calls.</remarks>
public sealed partial class Gpu : IDisposable
{
    /// <summary>The ILGPU context owning device discovery and compilation services.</summary>
    private readonly Context context;
    /// <summary>The initialized device used for this executor's allocations and kernel launches.</summary>
    private readonly Accelerator accelerator;
    /// <summary>Compiled indexed kernels cached by operation type.</summary>
    private readonly Dictionary<Type, Delegate> forKernels = new();
    /// <summary>Compiled element kernels cached by element type and operation type.</summary>
    private readonly Dictionary<(Type Element, Type Kernel), Delegate> forEachKernels = new();
    /// <summary>Outstanding allocations that must be released when the executor is disposed.</summary>
    private readonly HashSet<IDisposable> buffers = new();
    /// <summary>Whether the executor has released its context and accelerator.</summary>
    private bool disposed;

    /// <summary>Gets the lock that serializes executor operations and buffer access.</summary>
    internal object SyncRoot { get; } = new();

    /// <summary>Creates an executor and initializes an accelerator using the supplied options.</summary>
    /// <param name="options">Device selection options, or null to use the defaults.</param>
    /// <exception cref="ArgumentOutOfRangeException">The backend is invalid or the device index is unavailable.</exception>
    /// <exception cref="InvalidOperationException">No accelerator can be initialized under the requested options.</exception>
    public Gpu(GpuOptions? options = null)
    {
        options ??= new();
        if (!Enum.IsDefined(typeof(GpuBackend), options.Backend))
            throw new ArgumentOutOfRangeException(nameof(options), "Unknown backend.");
        if (options.DeviceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(options.DeviceIndex));

        context = Context.Create(builder =>
        {
            builder.CPU();
            if (options.Backend is GpuBackend.Auto or GpuBackend.Cuda)
                builder.Cuda();
            if (options.Backend is GpuBackend.Auto or GpuBackend.OpenCL)
                builder.OpenCL(device => (device.DeviceType & CLDeviceType.CL_DEVICE_TYPE_GPU) != 0);
        });
        try
        {
            accelerator = CreateAccelerator(context, options);
        }
        catch
        {
            context.Dispose();
            throw;
        }
    }

    /// <summary>Gets the display name of the initialized accelerator.</summary>
    public string DeviceName => accelerator.Name;
    /// <summary>Gets the backend of the initialized accelerator, after automatic selection or fallback.</summary>
    public GpuBackend Backend => accelerator.AcceleratorType switch
    {
        AcceleratorType.Cuda => GpuBackend.Cuda,
        AcceleratorType.OpenCL => GpuBackend.OpenCL,
        _ => GpuBackend.Cpu
    };
    /// <summary>Gets whether the initialized accelerator executes on a GPU.</summary>
    public bool IsGpu => Backend != GpuBackend.Cpu;

    /// <summary>Allocates a zero-filled device buffer owned by this executor.</summary>
    /// <typeparam name="T">The unmanaged, ILGPU-supported element type.</typeparam>
    /// <param name="length">The number of elements to allocate; zero is allowed.</param>
    /// <returns>A buffer that the caller can dispose independently of the executor.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The length is negative.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public GpuBuffer<T> Allocate<T>(int length) where T : unmanaged => AllocateCore<T>(length, clear: true);

    /// <summary>Allocates internal storage that will be filled completely before it can be read.</summary>
    /// <typeparam name="T">The unmanaged, ILGPU-supported element type.</typeparam>
    /// <param name="length">The number of elements to allocate.</param>
    /// <returns>An owned buffer with unspecified initial contents.</returns>
    internal GpuBuffer<T> AllocateUninitialized<T>(int length) where T : unmanaged =>
        AllocateCore<T>(length, clear: false);

    /// <summary>Allocates owned storage with optional initialization.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="length">The number of elements to allocate.</param>
    /// <param name="clear">Whether the public zero-initialization contract is required.</param>
    /// <returns>The newly owned buffer.</returns>
    private GpuBuffer<T> AllocateCore<T>(int length, bool clear) where T : unmanaged
    {
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            DeviceValueType.ValidateStorage(typeof(T));
            var memory = accelerator.Allocate1D<T>(length);
            try
            {
                if (clear && length > 0)
                    memory.MemSetToZero();
                var buffer = new GpuBuffer<T>(this, memory, length);
                buffers.Add(buffer);
                return buffer;
            }
            catch
            {
                memory.Dispose();
                throw;
            }
        }
    }

    /// <summary>Allocates a device buffer and copies a host array into it.</summary>
    /// <typeparam name="T">The unmanaged, ILGPU-supported element type.</typeparam>
    /// <param name="data">The host array to copy. The array is not retained by the buffer.</param>
    /// <returns>A device buffer containing the uploaded data.</returns>
    /// <exception cref="ArgumentNullException">The data array is null.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public GpuBuffer<T> Upload<T>(T[] data) where T : unmanaged
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        lock (SyncRoot)
        {
            var buffer = AllocateUninitialized<T>(data.Length);
            try
            {
                buffer.CopyFrom(data);
                return buffer;
            }
            catch
            {
                buffer.Dispose();
                throw;
            }
        }
    }

    /// <summary>Executes a value-type kernel for each index in a half-open range.</summary>
    /// <typeparam name="TKernel">A public value type implementing <see cref="IGpuForKernel"/>.</typeparam>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="kernel">The operation and its device-compatible state.</param>
    /// <remarks>Empty and reversed ranges do nothing. Iterations are unordered;
    /// shared writes require device synchronization or atomics.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations.</exception>
    /// <exception cref="ArgumentException">The kernel type is not publicly visible.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public void For<TKernel>(int fromInclusive, int toExclusive, TKernel kernel)
        where TKernel : struct, IGpuForKernel
    {
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            long count = (long)toExclusive - fromInclusive;
            if (count <= 0)
                return;
            if (count > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(toExclusive),
                    "One loop may contain at most Int32.MaxValue iterations.");

            if (!forKernels.TryGetValue(typeof(TKernel), out var compiled))
            {
                ValidateVisible(typeof(TKernel));
                compiled = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, TKernel>(
                    ExecuteFor<TKernel>);
                forKernels.Add(typeof(TKernel), compiled);
            }

            ((Action<Index1D, int, TKernel>)compiled)((int)count, fromInclusive, kernel);
            accelerator.Synchronize();
        }
    }

    /// <summary>Executes an operation over every element of a device buffer in place.</summary>
    /// <typeparam name="T">The public unmanaged, ILGPU-supported element type.</typeparam>
    /// <typeparam name="TKernel">A public value type implementing <see cref="IGpuForEachKernel{T}"/>.</typeparam>
    /// <param name="buffer">A live buffer owned by this executor.</param>
    /// <param name="kernel">The operation and its device-compatible state.</param>
    /// <remarks>The call waits for completion and does not copy the buffer back to the host.</remarks>
    /// <exception cref="ArgumentNullException">The buffer is null.</exception>
    /// <exception cref="ArgumentException">The buffer belongs to another executor, or a kernel or element type is not public.</exception>
    /// <exception cref="ObjectDisposedException">The executor or buffer has been disposed.</exception>
    public void ForEach<T, TKernel>(GpuBuffer<T> buffer, TKernel kernel)
        where T : unmanaged
        where TKernel : struct, IGpuForEachKernel<T>
    {
        if (buffer is null)
            throw new ArgumentNullException(nameof(buffer));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            buffer.ValidateOwner(this);
            if (buffer.Length == 0)
                return;

            var key = (typeof(T), typeof(TKernel));
            if (!forEachKernels.TryGetValue(key, out var compiled))
            {
                ValidateVisible(typeof(TKernel));
                ValidateVisible(typeof(T));
                compiled = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, TKernel>(
                    ExecuteForEach<T, TKernel>);
                forEachKernels.Add(key, compiled);
            }

            ((Action<Index1D, ArrayView<T>, TKernel>)compiled)(buffer.Length, buffer.View, kernel);
            accelerator.Synchronize();
        }
    }

    /// <summary>Uploads an array, processes its elements by reference, and copies results into the same array.</summary>
    /// <typeparam name="T">The public unmanaged, ILGPU-supported element type.</typeparam>
    /// <typeparam name="TKernel">A public value type implementing <see cref="IGpuForEachKernel{T}"/>.</typeparam>
    /// <param name="data">The array to process in place.</param>
    /// <param name="kernel">The operation and its device-compatible state.</param>
    /// <remarks>Use the buffer overload to retain data on the device between operations.</remarks>
    /// <exception cref="ArgumentNullException">The data array is null.</exception>
    /// <exception cref="ArgumentException">A kernel or element type is not publicly visible.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public void ForEach<T, TKernel>(T[] data, TKernel kernel)
        where T : unmanaged
        where TKernel : struct, IGpuForEachKernel<T>
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        lock (SyncRoot)
        {
            using var buffer = Upload(data);
            ForEach(buffer, kernel);
            buffer.CopyTo(data);
        }
    }

    /// <summary>Releases outstanding buffers, cached kernels, the accelerator, and its context.</summary>
    /// <remarks>Repeated calls have no effect. All buffer views become invalid after disposal.</remarks>
    public void Dispose()
    {
        lock (SyncRoot)
        {
            if (disposed)
                return;
            foreach (var buffer in buffers.ToArray())
                buffer.Dispose();
            lambdaKernels.Clear();
            forKernels.Clear();
            forEachKernels.Clear();
            reductionKernels.Clear();
            collectKernels.Clear();
            accelerator.Dispose();
            context.Dispose();
            disposed = true;
        }
    }

    /// <summary>Rejects operations after this executor has been disposed.</summary>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    internal void ThrowIfDisposed()
    {
        if (disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }
    /// <summary>Removes a released buffer from the executor's ownership registry.</summary>
    /// <param name="buffer">The buffer that has released its device allocation.</param>
    internal void Release(IDisposable buffer) => buffers.Remove(buffer);

    /// <summary>Rejects types that are not publicly visible to generated ILGPU kernel launchers.</summary>
    /// <param name="type">The kernel or element type to validate.</param>
    private static void ValidateVisible(Type type)
    {
        if (!type.IsVisible)
            throw new ArgumentException($"Kernel and element types must be public for ILGPU: {type}.");
    }

    /// <summary>Invokes a value-type operation with the original loop index.</summary>
    /// <typeparam name="TKernel">The device-compatible operation type.</typeparam>
    /// <param name="index">The device launch index.</param>
    /// <param name="start">The inclusive lower bound of the original range.</param>
    /// <param name="kernel">The operation and its state.</param>
    private static void ExecuteFor<TKernel>(Index1D index, int start, TKernel kernel)
        where TKernel : struct, IGpuForKernel => kernel.Execute(start + index.X);

    /// <summary>Invokes an operation with a reference to one device element.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <typeparam name="TKernel">The device-compatible operation type.</typeparam>
    /// <param name="index">The zero-based element index.</param>
    /// <param name="data">The device buffer view.</param>
    /// <param name="kernel">The operation and its state.</param>
    private static void ExecuteForEach<T, TKernel>(Index1D index, ArrayView<T> data, TKernel kernel)
        where T : unmanaged
        where TKernel : struct, IGpuForEachKernel<T> => kernel.Execute(index.X, ref data[index]);

    /// <summary>Initializes eligible devices in backend order and applies the configured CPU fallback.</summary>
    /// <param name="context">The ILGPU context containing the discovered devices.</param>
    /// <param name="options">Backend, device index, and fallback settings.</param>
    /// <returns>The first successfully initialized eligible accelerator.</returns>
    private static Accelerator CreateAccelerator(Context context, GpuOptions options)
    {
        var devices = context.Devices.Where(device => options.Backend switch
        {
            GpuBackend.Auto => device.AcceleratorType != AcceleratorType.CPU,
            GpuBackend.Cuda => device.AcceleratorType == AcceleratorType.Cuda,
            GpuBackend.OpenCL => device.AcceleratorType == AcceleratorType.OpenCL,
            GpuBackend.Cpu => device.AcceleratorType == AcceleratorType.CPU,
            _ => false
        }).OrderBy(device => device.AcceleratorType == AcceleratorType.Cuda ? 0 : 1).ToArray();

        if (options.DeviceIndex > 0 && options.DeviceIndex >= devices.Length)
            throw new ArgumentOutOfRangeException(nameof(options.DeviceIndex), "Device index is unavailable.");

        var failures = new List<Exception>();
        // Auto with index zero tries all GPUs; other selections try one device.
        var candidates = options.Backend == GpuBackend.Auto && options.DeviceIndex == 0
            ? devices
            : devices.Skip(options.DeviceIndex).Take(1);
        foreach (var device in candidates)
        {
            try { return device.CreateAccelerator(context); }
            catch (Exception exception) { failures.Add(exception); }
        }

        if (options.Backend == GpuBackend.Auto && options.AllowCpuFallback)
            return context.GetCPUDevice(0).CreateAccelerator(context);

        throw new InvalidOperationException(
            $"Cannot initialize backend {options.Backend}. Install a compatible GPU driver, " +
            "select a different backend/device, or explicitly select Cpu for debugging.",
            failures.Count > 0 ? new AggregateException(failures) : null);
    }
}
