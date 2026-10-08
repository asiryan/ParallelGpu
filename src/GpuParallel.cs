namespace ParallelGpu;

/// <summary>Provides GPU parallel loops with ordinary C# bodies and automatic collection transfers.</summary>
/// <remarks>Overloads without options reuse a lazily created default executor.
/// Overloads with options create an executor for each call; use <see cref="Gpu"/> to reuse
/// a configured device and its compiled kernels.</remarks>
public static class GpuParallel
{
    /// <summary>The process-wide executor, initialized only when a non-empty default loop runs.</summary>
    private static readonly Lazy<Gpu> Default = new(() => new Gpu());

    /// <summary>Registers process-exit cleanup for the shared default executor.</summary>
    static GpuParallel()
    {
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            if (Default.IsValueCreated)
                Default.Value.Dispose();
        };
    }

    /// <summary>Executes an ordinary loop body using the shared default GPU executor.</summary>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">The device-compatible loop body.</param>
    /// <remarks>Captured collections are transferred automatically. Results are available on return.
    /// Empty and reversed ranges do not initialize the default executor.</remarks>
    /// <exception cref="ArgumentNullException">The body is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations.</exception>
    /// <exception cref="NotSupportedException">The body or a captured collection is unsupported.</exception>
    /// <exception cref="InvalidOperationException">No default GPU accelerator can be initialized.</exception>
    public static void For(int fromInclusive, int toExclusive, Action<int> body)
    {
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        if (toExclusive <= fromInclusive)
            return;
        Default.Value.For(fromInclusive, toExclusive, body);
    }

    /// <summary>Executes a typed loop using the shared default GPU executor.</summary>
    /// <typeparam name="T">The ultimate element type of every captured collection: a supported primitive, enum, nullable value, or device-compatible structure without managed references.</typeparam>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">The device-compatible loop body.</param>
    /// <remarks>Use the non-generic overload for mixed collection types. Results are available on return.</remarks>
    /// <exception cref="ArgumentNullException">The body is null.</exception>
    /// <exception cref="ArgumentException">A captured collection has a different element type.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations.</exception>
    /// <exception cref="NotSupportedException">The element type, body, or a collection is unsupported.</exception>
    /// <exception cref="InvalidOperationException">No default GPU accelerator can be initialized.</exception>
    public static void For<T>(int fromInclusive, int toExclusive, Action<int> body)
    {
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        LambdaKernel.ValidateElementType(typeof(T));
        if (toExclusive <= fromInclusive)
            return;
        Default.Value.For<T>(fromInclusive, toExclusive, body);
    }

    /// <summary>Executes a loop using an executor created from the supplied device options.</summary>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="options">Backend, device index, and fallback settings.</param>
    /// <param name="body">The device-compatible loop body.</param>
    /// <remarks>The executor is disposed after execution. Empty and reversed ranges create no executor.</remarks>
    /// <exception cref="ArgumentNullException">The options or body is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The device selection or loop range is invalid.</exception>
    /// <exception cref="NotSupportedException">The body or a captured collection is unsupported.</exception>
    /// <exception cref="InvalidOperationException">No eligible accelerator can be initialized.</exception>
    public static void For(int fromInclusive, int toExclusive, GpuOptions options, Action<int> body)
    {
        if (options is null)
            throw new ArgumentNullException(nameof(options));
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        if (toExclusive <= fromInclusive)
            return;
        using var gpu = new Gpu(options);
        gpu.For(fromInclusive, toExclusive, body);
    }

    /// <summary>Executes a typed loop using an executor created from the supplied device options.</summary>
    /// <typeparam name="T">The ultimate element type of every captured collection: a supported primitive, enum, nullable value, or device-compatible structure without managed references.</typeparam>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="options">Backend, device index, and fallback settings.</param>
    /// <param name="body">The device-compatible loop body.</param>
    /// <remarks>The executor is disposed after execution. Use <see cref="Gpu"/> for repeated configured calls.</remarks>
    /// <exception cref="ArgumentNullException">The options or body is null.</exception>
    /// <exception cref="ArgumentException">A captured collection has a different element type.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The device selection or loop range is invalid.</exception>
    /// <exception cref="NotSupportedException">The element type, body, or a collection is unsupported.</exception>
    /// <exception cref="InvalidOperationException">No eligible accelerator can be initialized.</exception>
    public static void For<T>(int fromInclusive, int toExclusive, GpuOptions options, Action<int> body)
    {
        if (options is null)
            throw new ArgumentNullException(nameof(options));
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        LambdaKernel.ValidateElementType(typeof(T));
        if (toExclusive <= fromInclusive)
            return;
        using var gpu = new Gpu(options);
        gpu.For<T>(fromInclusive, toExclusive, body);
    }
}
