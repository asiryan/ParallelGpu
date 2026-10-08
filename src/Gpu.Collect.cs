using ILGPU;
using ILGPU.Runtime;

namespace ParallelGpu;

public sealed partial class Gpu
{
    /// <summary>Compiled bounded-output kernels cached by output and operation type.</summary>
    private readonly Dictionary<(Type Value, Type Kernel), Delegate> collectKernels = new();

    /// <summary>Allocates a reusable bounded output buffer with an initially empty result.</summary>
    /// <typeparam name="T">The unmanaged, ILGPU-supported output type.</typeparam>
    /// <param name="capacity">The maximum output length; zero is allowed.</param>
    /// <returns>An independently disposable output buffer owned by this executor.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The capacity is negative.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public GpuAppendBuffer<T> CreateAppendBuffer<T>(int capacity) where T : unmanaged
    {
        if (capacity < 0)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            var values = Allocate<T>(capacity);
            GpuBuffer<int>? state = null;
            GpuAppendBuffer<T>? output = null;
            try
            {
                state = Allocate<int>(2);
                output = new(this, values, state);
                buffers.Add(output);
                return output;
            }
            catch
            {
                output?.Dispose();
                state?.Dispose();
                values.Dispose();
                throw;
            }
        }
    }

    /// <summary>Runs indexed iterations that append zero or more values into bounded device output.</summary>
    /// <typeparam name="T">The public unmanaged, ILGPU-supported output type.</typeparam>
    /// <typeparam name="TKernel">A public value type implementing <see cref="IGpuCollectKernel{T}"/>.</typeparam>
    /// <param name="fromInclusive">The inclusive lower bound.</param>
    /// <param name="toExclusive">The exclusive upper bound.</param>
    /// <param name="capacity">The maximum number of output values.</param>
    /// <param name="kernel">The operation and its device-compatible state.</param>
    /// <returns>The produced values in unspecified iteration order.</returns>
    /// <remarks>Empty and reversed ranges return an empty result. Overflow rejects the entire result
    /// before any output values are copied to the host.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">The capacity is negative or the range is too large.</exception>
    /// <exception cref="ArgumentException">The output or operation type is not publicly visible.</exception>
    /// <exception cref="InvalidOperationException">The produced output exceeds capacity.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public T[] Collect<T, TKernel>(int fromInclusive, int toExclusive, int capacity, TKernel kernel)
        where T : unmanaged where TKernel : struct, IGpuCollectKernel<T>
    {
        if (capacity < 0)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            long count = (long)toExclusive - fromInclusive;
            if (count <= 0)
                return Array.Empty<T>();
            if (count > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(toExclusive),
                    "One loop may contain at most Int32.MaxValue iterations.");
            ValidateVisible(typeof(T));
            ValidateVisible(typeof(TKernel));
            var key = (typeof(T), typeof(TKernel));
            if (!collectKernels.TryGetValue(key, out var compiled))
            {
                compiled = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, GpuAppendView<T>, TKernel>(
                    ExecuteCollect<T, TKernel>);
                collectKernels.Add(key, compiled);
            }
            using var output = CreateAppendBuffer<T>(capacity);
            ((Action<Index1D, int, GpuAppendView<T>, TKernel>)compiled)((int)count, fromInclusive, output.View, kernel);
            accelerator.Synchronize();
            return output.ToArray();
        }
    }

    /// <summary>Invokes a bounded-output operation with the original index.</summary>
    /// <typeparam name="T">The output value type.</typeparam>
    /// <typeparam name="TKernel">The device-compatible operation.</typeparam>
    /// <param name="index">The launch index.</param>
    /// <param name="start">The original lower bound.</param>
    /// <param name="output">The bounded output view.</param>
    /// <param name="kernel">The operation and its state.</param>
    private static void ExecuteCollect<T, TKernel>(Index1D index, int start, GpuAppendView<T> output, TKernel kernel)
        where T : unmanaged where TKernel : struct, IGpuCollectKernel<T> =>
        kernel.Execute(start + index.X, output);
}
