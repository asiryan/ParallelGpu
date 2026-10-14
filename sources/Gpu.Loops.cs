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
    /// <para>Supports arrays (including nested arrays and nonzero lower bounds), lists, memory slices,
    /// and registered collection adapters. Elements may be supported primitives, enums, nullable values,
    /// or structures without managed references, explicit layout, pointer fields, inline arrays, or fixed
    /// buffers. Lists and contiguous views require value elements. Compatible constructors and helpers are supported.</para>
    /// <para>Captured scalars and collection references are read-only snapshots refreshed on each call;
    /// collection elements may be changed. Shared storage is preserved. Loop errors are checked before
    /// automatic host copies and adapter commits. Empty and reversed ranges do nothing.</para>
    /// <para>Iterations are unordered; shared writes require atomics or another synchronization scheme.
    /// Native device views must be live and owned by this executor. Their writes are not rolled back
    /// on failure; append and graph results require their own host validation.</para>
    /// <para>Requires a single ordinary <c>void(int)</c> delegate with inspectable IL and runtime code
    /// generation. Bound static delegates, DynamicMethod delegates, and Native AOT are unsupported.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">The body or a required constructor argument is null.</exception>
    /// <exception cref="ArgumentException">A capture, native view owner, or argument to a translated method is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range exceeds Int32.MaxValue iterations, or a translated collection or math operation receives an out-of-range argument.</exception>
    /// <exception cref="NotSupportedException">The body, value type, or collection uses an unsupported feature.</exception>
    /// <exception cref="ObjectDisposedException">The executor or a captured native view has been disposed.</exception>
    /// <exception cref="NullReferenceException">The loop accessed a null collection.</exception>
    /// <exception cref="IndexOutOfRangeException">An array, span, or adapted collection access is out of bounds.</exception>
    /// <exception cref="InvalidOperationException">The loop read an empty nullable, required storage from a null-backed ArraySegment, or wrote to a read-only collection.</exception>
    /// <exception cref="DivideByZeroException">Integer division or remainder used a zero divisor.</exception>
    /// <exception cref="OverflowException">Signed integer division, remainder, or Math.Abs overflowed.</exception>
    /// <exception cref="ArithmeticException">Math.Sign or MathF.Sign received NaN.</exception>
    public void For(int fromInclusive, int toExclusive, Action<int> body) =>
        ForBody(fromInclusive, toExclusive, body, null);

    /// <summary>Executes an ordinary loop body whose automatically transferred collections share one supported element type.</summary>
    /// <typeparam name="T">The element type shared by automatically transferred collections, including the innermost arrays of jagged arrays.</typeparam>
    /// <param name="fromInclusive">The inclusive lower bound of the range.</param>
    /// <param name="toExclusive">The exclusive upper bound of the range.</param>
    /// <param name="body">The device-compatible loop body.</param>
    /// <remarks>Execution rules and errors match <see cref="For(int, int, Action{int})"/>.
    /// Use that overload for mixed element types. This overload validates T even for empty or reversed ranges.</remarks>
    /// <inheritdoc cref="For(int, int, Action{int})" path="/exception[not(@cref='T:System.ArgumentException')]"/>
    /// <exception cref="ArgumentException">A captured collection has a different element type, or a capture, native view owner, or translated method argument is invalid.</exception>
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
