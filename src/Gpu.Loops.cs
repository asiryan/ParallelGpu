using System.Reflection;

namespace ParallelGpu;

public sealed partial class Gpu
{
    /// <summary>Compiled lambda kernels cached by body method; closure values are read separately on each call.</summary>
    private readonly Dictionary<MethodInfo, LambdaKernel> lambdaKernels = new();

    /// <summary>Compiles an ordinary C# loop body and executes it for each index in a half-open range.</summary>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">A loop body using supported values, collections, helpers, and GPU views.</param>
    /// <remarks>Collections include rectangular arrays of any CLR-supported rank, nested and mixed arrays,
    /// lists, memory slices, and registered adapters. Elements may be supported primitives, enums, nullable values,
    /// or structures without managed references. Aliases and array lower bounds are preserved.
    /// Captures are read on every invocation. Writable host collections are copied back only after a successful loop;
    /// null, bounds, nullable-value, slice, readonly, integer division or remainder, signed absolute value overflow,
    /// and inverted clamp bounds are reported before any automatic host collection copyback.
    /// Constructors and compatible helper methods accept values and collection parameters. Captured descriptors
    /// and scalar values are read-only. Explicit layouts and native pointer fields are unsupported.
    /// Iterations are unordered: shared mutations require synchronization such as GpuAtomic or explicit reduction.
    /// Captured append and graph views must be live and owned by this executor; their own deferred errors are checked
    /// by their host result APIs, and their device writes are not rolled back on a loop failure.
    /// Empty and reversed ranges do not compile or execute the body.</remarks>
    /// <exception cref="ArgumentNullException">The body is null, or an ArraySegment constructor uses a null array.</exception>
    /// <exception cref="ArgumentException">A captured native view belongs to another executor, or Math.Clamp used inverted bounds.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations, or the loop used an invalid list or ArraySegment index, or invalid slice.</exception>
    /// <exception cref="NotSupportedException">The body uses unsupported operations, including checked integer arithmetic or conversions, or a captured collection is unsupported.</exception>
    /// <exception cref="ObjectDisposedException">The executor or a captured native view has been disposed.</exception>
    /// <exception cref="NullReferenceException">The loop accessed a null collection.</exception>
    /// <exception cref="IndexOutOfRangeException">The loop accessed an invalid array or span index, array dimension, or index into a registered adapter snapshot.</exception>
    /// <exception cref="InvalidOperationException">The loop accessed an empty nullable, used a null-backed ArraySegment operation requiring storage, or wrote a read-only view.</exception>
    /// <exception cref="DivideByZeroException">Integer division or remainder used a zero divisor.</exception>
    /// <exception cref="OverflowException">Signed integer division or remainder of the minimum value by -1, or signed integer Math.Abs overflowed.</exception>
    public void For(int fromInclusive, int toExclusive, Action<int> body) =>
        ForBody(fromInclusive, toExclusive, body, null);

    /// <summary>Executes an ordinary loop body whose captured collections share one supported element type.</summary>
    /// <typeparam name="T">The element type of every captured collection: a supported primitive, enum, nullable value, or device-compatible structure without managed references.</typeparam>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">The device-compatible loop body.</param>
    /// <remarks>Use the non-generic overload for mixed collection element types.
    /// Compilation and transfers are automatic, and results are available on return.</remarks>
    /// <exception cref="ArgumentNullException">The body is null, or an ArraySegment constructor uses a null array.</exception>
    /// <exception cref="ArgumentException">A captured collection has a different element type, a captured native view belongs to another executor, or Math.Clamp used inverted bounds.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations, or the loop used an invalid list or ArraySegment index, or invalid slice.</exception>
    /// <exception cref="NotSupportedException">The element type, body, or captured collection is unsupported, including checked integer arithmetic or conversions.</exception>
    /// <exception cref="ObjectDisposedException">The executor or a captured native view has been disposed.</exception>
    /// <exception cref="NullReferenceException">The loop accessed a null collection.</exception>
    /// <exception cref="IndexOutOfRangeException">The loop accessed an invalid array or span index, array dimension, or index into a registered adapter snapshot.</exception>
    /// <exception cref="InvalidOperationException">The loop accessed an empty nullable, used a null-backed ArraySegment operation requiring storage, or wrote a read-only view.</exception>
    /// <exception cref="DivideByZeroException">Integer division or remainder used a zero divisor.</exception>
    /// <exception cref="OverflowException">Signed integer division or remainder of the minimum value by -1, or signed integer Math.Abs overflowed.</exception>
    public void For<T>(int fromInclusive, int toExclusive, Action<int> body)
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
