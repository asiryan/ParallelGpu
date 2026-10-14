using ILGPU;
using ILGPU.Runtime;

namespace ParallelGpu;

public sealed partial class Gpu
{
    /// <summary>Compiled reduction passes cached by value and operation type.</summary>
    private readonly Dictionary<(Type Value, Type Operation), Delegate> reductionKernels = new();

    /// <summary>Uploads an array and reduces its values using an associative operation.</summary>
    /// <typeparam name="T">The public unmanaged, ILGPU-supported element type.</typeparam>
    /// <typeparam name="TOperation">A public value type implementing <see cref="IGpuReduction{T}"/>.</typeparam>
    /// <param name="data">The host input array.</param>
    /// <param name="operation">The associative operation and its device-compatible state.</param>
    /// <returns>The ordered reduction, or the operation's identity for empty input.</returns>
    /// <remarks>The input remains unchanged. Empty input evaluates Identity on the host without
    /// compiling a device kernel; a single element is returned without calling Combine.
    /// Floating-point rounding follows a parallel tree and can differ from a sequential sum or
    /// between devices with different group sizes. Unsupported inline-array and fixed-buffer
    /// storage is rejected even when the input is empty.</remarks>
    /// <exception cref="ArgumentNullException">The input array is null.</exception>
    /// <exception cref="ArgumentException">An element or operation type is not publicly visible.</exception>
    /// <exception cref="NotSupportedException">The input type contains inline arrays or fixed-buffer storage,
    /// or a value cannot fit in device shared memory.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public T Reduce<T, TOperation>(T[] data, TOperation operation)
        where T : unmanaged where TOperation : struct, IGpuReduction<T>
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            ValidateVisible(typeof(T));
            ValidateVisible(typeof(TOperation));
            DeviceValueType.ValidateStorage(typeof(T));
            if (data.Length == 0)
                return operation.Identity;
            using var buffer = Upload(data);
            return Reduce(buffer, operation);
        }
    }

    /// <summary>Reduces device-resident values without downloading the input.</summary>
    /// <typeparam name="T">The public unmanaged, ILGPU-supported element type.</typeparam>
    /// <typeparam name="TOperation">A public value type implementing <see cref="IGpuReduction{T}"/>.</typeparam>
    /// <param name="buffer">A live input buffer owned by this executor.</param>
    /// <param name="operation">The associative operation and its device-compatible state.</param>
    /// <returns>The ordered reduction, or the operation's identity for empty input.</returns>
    /// <remarks>Only the final value is copied to the host. Input order is preserved, but parallel
    /// grouping can change floating-point rounding. Empty input evaluates Identity on the host;
    /// a single element is returned without calling Combine. Both cases skip kernel compilation.</remarks>
    /// <exception cref="ArgumentNullException">The input buffer is null.</exception>
    /// <exception cref="ArgumentException">The buffer belongs to another executor or a type is not public.</exception>
    /// <exception cref="NotSupportedException">A value cannot fit in device shared memory.</exception>
    /// <exception cref="ObjectDisposedException">The executor or buffer has been disposed.</exception>
    public T Reduce<T, TOperation>(GpuBuffer<T> buffer, TOperation operation)
        where T : unmanaged where TOperation : struct, IGpuReduction<T>
    {
        if (buffer is null)
            throw new ArgumentNullException(nameof(buffer));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            buffer.ValidateOwner(this);
            ValidateVisible(typeof(T));
            ValidateVisible(typeof(TOperation));
            if (buffer.Length == 0)
                return operation.Identity;
            if (buffer.Length == 1)
                return buffer.ToArray()[0];

            int limit = Math.Min(128, Math.Min(accelerator.MaxNumThreadsPerGroup,
                accelerator.MaxSharedMemoryPerGroup / Interop.SizeOf<T>()));
            if (limit < 1)
                throw new NotSupportedException("The reduction value cannot fit in device shared memory.");
            int groupSize = 1;
            while (groupSize <= limit / 2)
                groupSize *= 2;

            var key = (typeof(T), typeof(TOperation));
            if (!reductionKernels.TryGetValue(key, out var compiled))
            {
                compiled = accelerator.LoadStreamKernel<ArrayView<T>, ArrayView<T>, int, TOperation>(
                    ExecuteReductionPass<T, TOperation>);
                reductionKernels.Add(key, compiled);
            }
            var launch = (Action<KernelConfig, ArrayView<T>, ArrayView<T>, int, TOperation>)compiled;
            int chunkSize = groupSize * 2;
            int firstLength = (buffer.Length - 1) / chunkSize + 1;
            using var first = AllocateUninitialized<T>(firstLength);
            using var second = AllocateUninitialized<T>((firstLength - 1) / chunkSize + 1);
            var input = buffer.View;
            int length = buffer.Length;
            bool useFirst = true;
            while (length > 1)
            {
                int outputLength = (length - 1) / chunkSize + 1;
                var output = useFirst ? first.View : second.View;
                int passGroupSize = groupSize;
                while (passGroupSize > 1 && length <= passGroupSize)
                    passGroupSize /= 2;
                var configuration = new KernelConfig(outputLength, passGroupSize,
                    SharedMemoryConfig.RequestDynamic<T>(passGroupSize));
                launch(configuration, input, output, length, operation);
                input = output;
                length = outputLength;
                useFirst = !useFirst;
            }
            accelerator.Synchronize();
            var result = new T[1];
            (useFirst ? second : first).CopyTo(0, result);
            return result[0];
        }
    }

    /// <summary>Reduces consecutive pairs and then their adjacent shared-memory segments.</summary>
    /// <typeparam name="T">The device value type.</typeparam>
    /// <typeparam name="TOperation">The associative device operation.</typeparam>
    /// <param name="input">The input values of this pass.</param>
    /// <param name="output">One result per work group.</param>
    /// <param name="length">The valid input length.</param>
    /// <param name="operation">The reduction operation.</param>
    private static void ExecuteReductionPass<T, TOperation>(ArrayView<T> input, ArrayView<T> output,
        int length, TOperation operation)
        where T : unmanaged where TOperation : struct, IGpuReduction<T>
    {
        int lane = Group.IdxX;
        int index = Grid.IdxX * (Group.DimX * 2) + lane * 2;
        var shared = SharedMemory.GetDynamic<T>();
        T value = operation.Identity;
        if (index < length)
        {
            value = input[index];
            if (index + 1 < length)
                value = operation.Combine(value, input[index + 1]);
        }
        shared[lane] = value;
        Group.Barrier();
        for (int stride = 1; stride < Group.DimX; stride *= 2)
        {
            if ((lane & (stride * 2 - 1)) == 0)
                shared[lane] = operation.Combine(shared[lane], shared[lane + stride]);
            Group.Barrier();
        }
        if (lane == 0)
            output[Grid.IdxX] = shared[0];
    }
}
