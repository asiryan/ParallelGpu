using System.Reflection;

namespace ParallelGpu;

public sealed partial class GpuParallel
{
    private readonly Dictionary<MethodInfo, LambdaKernel> lambdaKernels = new();

    /// <summary>Compiles an ordinary loop body and automatically transfers its captured collections.</summary>
    /// <remarks>Supports numeric T[], T[,], T[][] and List&lt;T&gt;, including IList&lt;T&gt;
    /// and IReadOnlyList&lt;T&gt; references to them. Calls return after copying results back.</remarks>
    public void For(int fromInclusive, int toExclusive, Action<int> body) =>
        ForBody(fromInclusive, toExclusive, body, null);

    /// <summary>Executes a loop whose captured collections all have the numeric element type T.</summary>
    /// <remarks>Use the non-generic overload when mixing collection element types.</remarks>
    public void For<T>(int fromInclusive, int toExclusive, Action<int> body) where T : unmanaged
    {
        LambdaKernel.ValidateElementType(typeof(T));
        ForBody(fromInclusive, toExclusive, body, typeof(T));
    }

    private void ForBody(int fromInclusive, int toExclusive, Action<int> body, Type? elementType)
    {
        if (body is null)
            throw new ArgumentNullException(nameof(body));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            long count = (long)toExclusive - fromInclusive;
            if (count <= 0)
                return;
            if (count > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(toExclusive),
                    "One loop may contain at most Int32.MaxValue iterations.");
            if (body.GetInvocationList().Length != 1)
                throw new NotSupportedException("Multicast delegates are not supported as GPU loop bodies.");
            if (!lambdaKernels.TryGetValue(body.Method, out var compiled))
            {
                compiled = LambdaKernel.Compile(accelerator, body.Method);
                lambdaKernels.Add(body.Method, compiled);
            }
            compiled.Execute(this, body, fromInclusive, (int)count, elementType);
        }
    }

    internal void Synchronize() => accelerator.Synchronize();
}

