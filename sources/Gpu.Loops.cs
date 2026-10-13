using System.Reflection;

namespace ParallelGpu;

public sealed partial class Gpu
{
    /// <summary>Compiled lambda kernels cached by body method; closure values are read separately on each call.</summary>
    private readonly Dictionary<MethodInfo, LambdaKernel> lambdaKernels = new();
    /// <summary>The shared deferred-error slot for automatic loops serialized by SyncRoot.</summary>
    private GpuBuffer<int>? loopErrors;

    /// <summary>Resets the executor's reusable error slot before one automatic-loop invocation.</summary>
    /// <returns>The owned single-element buffer, released when the executor is disposed.</returns>
    /// <remarks>The caller holds SyncRoot until kernel completion and error validation.</remarks>
    internal GpuBuffer<int> PrepareLoopErrors()
    {
        if (loopErrors is null)
            loopErrors = Allocate<int>(1);
        else
            loopErrors.Clear();
        return loopErrors;
    }

    /// <summary>Compiles an ordinary C# loop body and executes it for each index in a half-open range.</summary>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">A loop body using supported values, collections, helpers, and GPU views.</param>
    /// <remarks>
    /// <para>Collections include rectangular arrays of any CLR-supported rank, nested and mixed arrays,
    /// lists, memory slices, and registered adapters. Elements may be supported primitives, enums, nullable values,
    /// or structures without managed references. Aliases and array lower bounds are preserved.
    /// Lists and contiguous views require value elements; access nested arrays directly.
    /// Constructors and compatible helpers accept values and collections. Captured descriptors and scalar values
    /// are read-only. Explicit layouts and native pointer fields are unsupported.</para>
    /// <para>Captures, including volatile fields, use host snapshots refreshed on every call.
    /// Deferred loop errors are checked before automatic host
    /// copyback and adapter commits. Empty and reversed ranges skip compilation and execution.</para>
    /// <para>Iterations are unordered; shared writes require synchronization such as atomics or a reduction.
    /// Captured buffer, append, and graph views must be live and owned by this executor. Their device writes
    /// are not rolled back on a loop failure; append and graph result APIs check their own deferred errors.</para>
    /// <para>Requires an ordinary void(int) method with inspectable IL and runtime code generation.
    /// Closed static delegates with bound arguments, DynamicMethod delegates, and Native AOT are unsupported.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">The body is null, or an ArraySegment constructor receives a null array.</exception>
    /// <exception cref="ArgumentException">An intermediate capture object is null, a captured native view belongs to another executor, Math.Clamp used inverted bounds, a rounding mode is invalid, or an ArraySegment constructor range exceeds its array.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations, or the loop used invalid rounding digits, an invalid list or ArraySegment index, invalid slice, or negative ArraySegment constructor offset or count.</exception>
    /// <exception cref="NotSupportedException">The body uses unsupported operations, including checked integer arithmetic or conversions, or a captured collection is unsupported.</exception>
    /// <exception cref="ObjectDisposedException">The executor or a captured native view has been disposed.</exception>
    /// <exception cref="NullReferenceException">The loop accessed a null collection.</exception>
    /// <exception cref="IndexOutOfRangeException">The loop accessed an invalid array or span index, array dimension, or index into a registered adapter snapshot.</exception>
    /// <exception cref="InvalidOperationException">The loop accessed an empty nullable, used a null-backed ArraySegment operation requiring storage, or wrote a read-only view.</exception>
    /// <exception cref="DivideByZeroException">Integer division or remainder used a zero divisor.</exception>
    /// <exception cref="OverflowException">Signed integer division or remainder of the minimum value by -1, or signed integer Math.Abs overflowed.</exception>
    /// <exception cref="ArithmeticException">Math.Sign or MathF.Sign received NaN.</exception>
    public void For(int fromInclusive, int toExclusive, Action<int> body) =>
        ForBody(fromInclusive, toExclusive, body, null);

    /// <summary>Executes an ordinary loop body whose captured collections share one supported element type.</summary>
    /// <typeparam name="T">The ultimate element type of every captured collection: a supported primitive, enum, nullable value, or device-compatible structure without managed references.</typeparam>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">The device-compatible loop body.</param>
    /// <remarks>Use the non-generic overload for mixed collection element types.
    /// Compilation and transfers are automatic, and results are available on return.
    /// The element type is validated even for empty and reversed ranges; other execution rules match the non-generic overload.</remarks>
    /// <exception cref="ArgumentNullException">The body is null, or an ArraySegment constructor receives a null array.</exception>
    /// <exception cref="ArgumentException">A captured collection has a different element type, an intermediate capture object is null, a captured native view belongs to another executor, Math.Clamp used inverted bounds, a rounding mode is invalid, or an ArraySegment constructor range exceeds its array.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range contains more than Int32.MaxValue iterations, or the loop used invalid rounding digits, an invalid list or ArraySegment index, invalid slice, or negative ArraySegment constructor offset or count.</exception>
    /// <exception cref="NotSupportedException">The element type, body, or captured collection is unsupported, including checked integer arithmetic or conversions.</exception>
    /// <exception cref="ObjectDisposedException">The executor or a captured native view has been disposed.</exception>
    /// <exception cref="NullReferenceException">The loop accessed a null collection.</exception>
    /// <exception cref="IndexOutOfRangeException">The loop accessed an invalid array or span index, array dimension, or index into a registered adapter snapshot.</exception>
    /// <exception cref="InvalidOperationException">The loop accessed an empty nullable, used a null-backed ArraySegment operation requiring storage, or wrote a read-only view.</exception>
    /// <exception cref="DivideByZeroException">Integer division or remainder used a zero divisor.</exception>
    /// <exception cref="OverflowException">Signed integer division or remainder of the minimum value by -1, or signed integer Math.Abs overflowed.</exception>
    /// <exception cref="ArithmeticException">Math.Sign or MathF.Sign received NaN.</exception>
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
