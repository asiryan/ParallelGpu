using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace ParallelGpu;

/// <summary>Owns an ILGPU device and executes synchronous parallel loops.</summary>
/// <remarks>Calls on one instance are serialized. Each call waits for completion,
/// so its buffers can be read or disposed immediately after it returns.</remarks>
public sealed partial class GpuParallel : IDisposable
{
    private readonly Context context;
    private readonly Accelerator accelerator;
    private readonly Dictionary<Type, Delegate> forKernels = new();
    private readonly Dictionary<(Type Element, Type Kernel), Delegate> forEachKernels = new();
    private readonly HashSet<IDisposable> buffers = new();
    private bool disposed;

    internal object SyncRoot { get; } = new();

    public GpuParallel(GpuParallelOptions? options = null)
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

    public string DeviceName => accelerator.Name;
    public GpuBackend Backend => accelerator.AcceleratorType switch
    {
        AcceleratorType.Cuda => GpuBackend.Cuda,
        AcceleratorType.OpenCL => GpuBackend.OpenCL,
        _ => GpuBackend.Cpu
    };
    public bool IsGpu => Backend != GpuBackend.Cpu;

    /// <summary>Allocates a zero-filled buffer owned by this instance.</summary>
    public GpuBuffer<T> Allocate<T>(int length) where T : unmanaged
    {
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            var memory = accelerator.Allocate1D<T>(length);
            try
            {
                if (length > 0)
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

    /// <summary>Copies an array into a new GPU buffer. The array is not retained.</summary>
    public GpuBuffer<T> Upload<T>(T[] data) where T : unmanaged
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        lock (SyncRoot)
        {
            var buffer = Allocate<T>(data.Length);
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

    /// <summary>Executes [fromInclusive, toExclusive). An empty or reversed range
    /// is a no-op, as in System.Threading.Tasks.Parallel.For.</summary>
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

    /// <summary>Executes an operation over every element of a buffer in place.</summary>
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

    /// <summary>Convenience overload: uploads, processes and copies back into the
    /// same array. Use the buffer overload to avoid transfers between loops.</summary>
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
            accelerator.Dispose();
            context.Dispose();
            disposed = true;
        }
    }

    internal void ThrowIfDisposed()
    {
        if (disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }
    internal void Release(IDisposable buffer) => buffers.Remove(buffer);

    private static void ValidateVisible(Type type)
    {
        if (!type.IsVisible)
            throw new ArgumentException($"Kernel and element types must be public for ILGPU: {type}.");
    }

    private static void ExecuteFor<TKernel>(Index1D index, int start, TKernel kernel)
        where TKernel : struct, IGpuForKernel => kernel.Execute(start + index.X);

    private static void ExecuteForEach<T, TKernel>(Index1D index, ArrayView<T> data, TKernel kernel)
        where T : unmanaged
        where TKernel : struct, IGpuForEachKernel<T> => kernel.Execute(index.X, ref data[index]);

    private static Accelerator CreateAccelerator(Context context, GpuParallelOptions options)
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

