using ILGPU;

namespace ParallelGpu;

/// <summary>Carries deferred errors so failed loops never copy partial results to host collections.</summary>
internal readonly struct DeviceExecutionContext
{
    /// <summary>The first deferred error code, initialized to zero for each invocation.</summary>
    public readonly ArrayView<int> Errors;

    /// <summary>Creates the context for a single invocation.</summary>
    /// <param name="errors">The single-element error buffer.</param>
    /// <param name="scratch">Per-iteration storage used after an invalid ref access.</param>
    /// <param name="scratchStride">Bytes allocated for each iteration.</param>
    /// <param name="threadIndex">The zero-based iteration owning this context.</param>
    public DeviceExecutionContext(ArrayView<int> errors, ArrayView<byte> scratch = default, int scratchStride = 0, int threadIndex = 0)
    {
        Errors = errors;
        Scratch = scratch;
        ScratchStride = scratchStride;
        ThreadIndex = threadIndex;
    }

    public readonly ArrayView<byte> Scratch;
    public readonly int ScratchStride;
    public readonly int ThreadIndex;

    public static DeviceExecutionContext WithThread(DeviceExecutionContext context, int threadIndex) =>
        new(context.Errors, context.Scratch, context.ScratchStride, threadIndex);

    public ref T ScratchAddress<T>() where T : unmanaged =>
        ref Scratch.SubView((long)ThreadIndex * ScratchStride, ScratchStride).Cast<T>()[0];

    /// <summary>Records the first failure without throwing inside a device kernel.</summary>
    /// <param name="code">One for an empty nullable, two for a null collection, three for an invalid index.</param>
    public void Fail(int code)
    {
        if (Errors.Length > 0)
            Atomic.CompareExchange(ref Errors[0], 0, code);
    }

    /// <summary>Throws the host equivalent of a deferred device failure.</summary>
    /// <param name="code">The code downloaded after synchronization.</param>
    internal static void ThrowIfFailed(int code)
    {
        switch (code)
        {
            case 0: return;
            case 1: throw new InvalidOperationException("Nullable object must have a value.");
            case 2: throw new NullReferenceException("A GPU loop accessed a null collection.");
            case 3: throw new IndexOutOfRangeException("A GPU loop index was outside the collection bounds.");
            case 5: throw new InvalidOperationException("A GPU loop attempted to write a read-only collection.");
            case 6: throw new ArgumentOutOfRangeException("slice", "A GPU collection slice was outside the collection bounds.");
            default: throw new InvalidOperationException("A GPU loop reported an invalid device operation.");
        }
    }
}
