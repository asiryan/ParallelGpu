namespace ParallelGpu;

/// <summary>GPU loops with ordinary C# bodies and automatic collection transfers.</summary>
public static class Parallel
{
    private static readonly Lazy<GpuParallel> Default = new(() => new GpuParallel());

    static Parallel()
    {
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            if (Default.IsValueCreated)
                Default.Value.Dispose();
        };
    }

    /// <summary>Executes a loop using the default GPU, compiling and caching its body.</summary>
    public static void For(int fromInclusive, int toExclusive, Action<int> body)
    {
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        if (toExclusive <= fromInclusive)
            return;
        Default.Value.For(fromInclusive, toExclusive, body);
    }

    /// <summary>Executes a loop whose captured collections contain elements of T.</summary>
    public static void For<T>(int fromInclusive, int toExclusive, Action<int> body) where T : unmanaged
    {
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        LambdaKernel.ValidateElementType(typeof(T));
        if (toExclusive <= fromInclusive)
            return;
        Default.Value.For<T>(fromInclusive, toExclusive, body);
    }

    /// <summary>Executes a loop on the selected backend. Use GpuParallel to reuse a device.</summary>
    public static void For(int fromInclusive, int toExclusive, GpuParallelOptions options, Action<int> body)
    {
        if (options is null)
            throw new ArgumentNullException(nameof(options));
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        if (toExclusive <= fromInclusive)
            return;
        using var gpu = new GpuParallel(options);
        gpu.For(fromInclusive, toExclusive, body);
    }

    /// <summary>Executes a typed loop on the selected backend.</summary>
    public static void For<T>(int fromInclusive, int toExclusive, GpuParallelOptions options, Action<int> body)
        where T : unmanaged
    {
        if (options is null)
            throw new ArgumentNullException(nameof(options));
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        LambdaKernel.ValidateElementType(typeof(T));
        if (toExclusive <= fromInclusive)
            return;
        using var gpu = new GpuParallel(options);
        gpu.For<T>(fromInclusive, toExclusive, body);
    }
}
