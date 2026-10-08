using System.Reflection;

namespace ParallelGpu;

public sealed partial class Gpu
{
    /// <summary>Compiled lambda kernels cached by body method; closure values are read separately on each call.</summary>
    private readonly Dictionary<MethodInfo, LambdaKernel> lambdaKernels = new();

    /// <summary>Compiles an ordinary C# loop body and executes it for each index in a half-open range.</summary>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">A loop body using supported captured arrays and lists.</param>
    /// <remarks>Supported collections include flat, rectangular, and jagged arrays and lists of numeric, Boolean, character, or enum elements.
    /// Captured collections are uploaded automatically; collections written to or exposed by reference
    /// are copied back before return. Aliases are preserved. Iterations are unordered and must be independent.
    /// Empty and reversed ranges do not compile or execute the body.</remarks>
    /// <exception cref="ArgumentNullException">The body is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations.</exception>
    /// <exception cref="NotSupportedException">The body or a captured collection is not supported by the device compiler.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public void For(int fromInclusive, int toExclusive, Action<int> body) =>
        ForBody(fromInclusive, toExclusive, body, null);

    /// <summary>Executes an ordinary loop body whose captured collections share one supported element type.</summary>
    /// <typeparam name="T">The element type of every captured collection: float, int, uint, long, ulong, double, short, ushort, byte, sbyte, bool, char, or an enum.</typeparam>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">The device-compatible loop body.</param>
    /// <remarks>Use the non-generic overload for mixed collection element types.
    /// Compilation and transfers are automatic, and results are available on return.</remarks>
    /// <exception cref="ArgumentNullException">The body is null.</exception>
    /// <exception cref="ArgumentException">A captured collection has a different element type.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations.</exception>
    /// <exception cref="NotSupportedException">The element type, body, or a captured collection is unsupported.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public void For<T>(int fromInclusive, int toExclusive, Action<int> body) where T : unmanaged
    {
        LambdaKernel.ValidateElementType(typeof(T));
        ForBody(fromInclusive, toExclusive, body, typeof(T));
    }

    /// <summary>Validates a loop range, retrieves its compiled kernel, and executes the current closure.</summary>
    /// <param name="fromInclusive">The inclusive lower bound.</param>
    /// <param name="toExclusive">The exclusive upper bound.</param>
    /// <param name="body">The single-cast loop delegate.</param>
    /// <param name="elementType">The required collection element type, or null for mixed types.</param>
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
            // Cache the method's translated instructions; Execute reads the current delegate's closure values.
            if (!lambdaKernels.TryGetValue(body.Method, out var compiled))
            {
                compiled = LambdaKernel.Compile(accelerator, body.Method);
                lambdaKernels.Add(body.Method, compiled);
            }
            compiled.Execute(this, body, fromInclusive, (int)count, elementType);
        }
    }

    /// <summary>Waits for all pending work on the accelerator to complete.</summary>
    internal void Synchronize() => accelerator.Synchronize();
}

