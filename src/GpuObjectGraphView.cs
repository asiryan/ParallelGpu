using ILGPU;

namespace ParallelGpu;

/// <summary>A device view over bounded object records, outgoing references, and graph roots.</summary>
/// <typeparam name="TRecord">The unmanaged, ILGPU-supported payload type.</typeparam>
/// <remarks>Null, shared references, and cycles use stable graph-specific handles. Existing records
/// and links may be edited; concurrent creation reserves distinct records and reference ranges. New objects must be
/// read only in a later kernel, after their creating kernel has completed. Concurrent changes to
/// the same payload, reference slot, or root still require synchronization by the caller.
/// Invalid references and capacity exhaustion are recorded and rejected on host download. Lambda
/// captures validate ownership and lifetime before launch. Views in explicit value kernels must
/// not outlive their graph and must be used only by its owning executor.</remarks>
public readonly struct GpuObjectGraphView<TRecord> : IDeviceNativeView where TRecord : unmanaged
{
    /// <summary>The host-validated allocation lifetime.</summary>
    private readonly int lifetimeId;
    /// <summary>The unique graph identity.</summary>
    private readonly int graphId;
    /// <summary>The bounded record pool.</summary>
    private readonly ArrayView<TRecord> values;
    /// <summary>The first reference slot of each record.</summary>
    private readonly ArrayView<int> offsets;
    /// <summary>The fixed outgoing-reference count of each record.</summary>
    private readonly ArrayView<int> lengths;
    /// <summary>The bounded pool of one-based target record indices.</summary>
    private readonly ArrayView<int> references;
    /// <summary>The graph roots, stored as one-based indices or zero.</summary>
    private readonly ArrayView<int> roots;
    /// <summary>The allocated record count, reference count, overflow flag, and invalid-access flag.</summary>
    private readonly ArrayView<int> state;

    /// <summary>Builds a view over host-validated allocations.</summary>
    /// <param name="lifetimeId">The allocation lifetime identity.</param>
    /// <param name="graphId">The unique graph identity.</param>
    /// <param name="values">The record pool.</param>
    /// <param name="offsets">The per-record reference offsets.</param>
    /// <param name="lengths">The per-record reference counts.</param>
    /// <param name="references">The reference pool.</param>
    /// <param name="roots">The roots.</param>
    /// <param name="state">The allocation counters and failure flags.</param>
    internal GpuObjectGraphView(int lifetimeId, int graphId, ArrayView<TRecord> values, ArrayView<int> offsets,
        ArrayView<int> lengths, ArrayView<int> references, ArrayView<int> roots, ArrayView<int> state)
    {
        this.lifetimeId = lifetimeId;
        this.graphId = graphId;
        this.values = values;
        this.offsets = offsets;
        this.lengths = lengths;
        this.references = references;
        this.roots = roots;
        this.state = state;
    }

    /// <summary>Validates this capture before the host launches a lambda kernel.</summary>
    /// <param name="owner">The launching executor.</param>
    void IDeviceNativeView.ValidateOwner(Gpu owner) => GpuViewLifetime.Validate(lifetimeId, owner);

    /// <summary>Gets the maximum number of object records.</summary>
    public int Capacity => values.IntLength;

    /// <summary>Gets the maximum total number of outgoing reference slots.</summary>
    public int ReferenceCapacity => references.IntLength;

    /// <summary>Gets the fixed number of graph roots.</summary>
    public int RootCount => roots.IntLength;

    /// <summary>Reads a graph root.</summary>
    /// <param name="index">The zero-based root index.</param>
    /// <returns>The root reference, or null on invalid access.</returns>
    public GpuObjectHandle GetRoot(int index)
    {
        if (index < 0 || index >= roots.IntLength)
        {
            Fail();
            return default;
        }
        return Handle(roots[index]);
    }

    /// <summary>Changes a graph root to an existing or newly created object, or null.</summary>
    /// <param name="index">The zero-based root index.</param>
    /// <param name="target">The target reference belonging to this graph, or null.</param>
    public void SetRoot(int index, GpuObjectHandle target)
    {
        if (index < 0 || index >= roots.IntLength || !ValidateTarget(target))
        {
            Fail();
            return;
        }
        roots[index] = target.Value;
    }

    /// <summary>Reads the payload of a non-null object.</summary>
    /// <param name="handle">The graph-specific object reference.</param>
    /// <returns>The payload, or its default value on invalid access.</returns>
    public TRecord Read(GpuObjectHandle handle) => ValidateObject(handle) ? values[handle.Index] : default;

    /// <summary>Replaces the payload of a non-null object.</summary>
    /// <param name="handle">The graph-specific object reference.</param>
    /// <param name="value">The replacement payload.</param>
    public void Write(GpuObjectHandle handle, TRecord value)
    {
        if (ValidateObject(handle))
            values[handle.Index] = value;
    }

    /// <summary>Gets the fixed number of outgoing references of a non-null object.</summary>
    /// <param name="handle">The graph-specific object reference.</param>
    /// <returns>The outgoing-reference count, or zero on invalid access.</returns>
    public int GetReferenceCount(GpuObjectHandle handle) => ValidateObject(handle) ? lengths[handle.Index] : 0;

    /// <summary>Reads one outgoing reference of an object.</summary>
    /// <param name="handle">The non-null source object.</param>
    /// <param name="index">The zero-based outgoing-reference index.</param>
    /// <returns>The target reference, or null on invalid access.</returns>
    public GpuObjectHandle GetReference(GpuObjectHandle handle, int index)
    {
        if (!ValidateSlot(handle, index))
            return default;
        return Handle(references[offsets[handle.Index] + index]);
    }

    /// <summary>Changes an outgoing reference to another object or null.</summary>
    /// <param name="handle">The non-null source object.</param>
    /// <param name="index">The zero-based outgoing-reference index.</param>
    /// <param name="target">The graph-specific target object, or null.</param>
    public void SetReference(GpuObjectHandle handle, int index, GpuObjectHandle target)
    {
        if (!ValidateSlot(handle, index) || !ValidateTarget(target))
            return;
        references[offsets[handle.Index] + index] = target.Value;
    }

    /// <summary>Creates a bounded object record with initially null outgoing references.</summary>
    /// <param name="value">The new payload.</param>
    /// <param name="referenceCount">The fixed number of outgoing references to reserve.</param>
    /// <returns>The new object reference, or null when allocation fails.</returns>
    /// <remarks>Capacity exhaustion marks the complete result as overflowed; no host reconstruction
    /// is allowed. Reference counts cannot be resized, but links can be changed with SetReference.</remarks>
    public GpuObjectHandle Create(TRecord value, int referenceCount) =>
        TryCreate(value, referenceCount, out var handle) ? handle : default;

    /// <summary>Attempts to reserve and initialize a record with fixed outgoing-reference slots.</summary>
    /// <param name="value">The new payload.</param>
    /// <param name="referenceCount">The nonnegative number of outgoing references.</param>
    /// <param name="handle">The newly allocated reference, or null when allocation fails.</param>
    /// <returns>True when the complete record was initialized; otherwise false.</returns>
    /// <remarks>Record and reference reservations use separate atomics. On failure, reservations are
    /// not rolled back and host download rejects the graph.</remarks>
    public bool TryCreate(TRecord value, int referenceCount, out GpuObjectHandle handle)
    {
        handle = default;
        if (referenceCount < 0)
        {
            Fail();
            return false;
        }
        int record = Reserve(0, 1, values.IntLength);
        if (record < 0)
            return false;
        int firstReference = Reserve(1, referenceCount, references.IntLength);
        if (firstReference < 0)
            return false;
        values[record] = value;
        offsets[record] = firstReference;
        lengths[record] = referenceCount;
        for (int i = 0; i < referenceCount; i++)
            references[firstReference + i] = 0;
        handle = new GpuObjectHandle(graphId, record + 1);
        return true;
    }

    /// <summary>Reserves a saturating bounded range without arithmetic overflow.</summary>
    /// <param name="counter">The allocation counter index.</param>
    /// <param name="amount">The nonnegative range length.</param>
    /// <param name="capacity">The pool's capacity.</param>
    /// <returns>The reserved range start, or -1 on exhaustion.</returns>
    private int Reserve(int counter, int amount, int capacity)
    {
        int current = Atomic.Add(ref state[counter], 0);
        if (amount == 0)
            return current;
        while (amount <= capacity && current <= capacity - amount)
        {
            int observed = Atomic.CompareExchange(ref state[counter], current, current + amount);
            if (observed == current)
                return current;
            current = observed;
        }
        Atomic.Exchange(ref state[2], 1);
        return -1;
    }

    /// <summary>Builds a nullable handle from a stored target index.</summary>
    /// <param name="value">The one-based target index.</param>
    /// <returns>The graph-specific handle or null.</returns>
    private GpuObjectHandle Handle(int value) => value == 0 ? default : new GpuObjectHandle(graphId, value);

    /// <summary>Validates a nullable target and records invalid access.</summary>
    /// <param name="handle">The target reference.</param>
    /// <returns>True when it is null or belongs to the current allocated pool.</returns>
    private bool ValidateTarget(GpuObjectHandle handle)
    {
        if (handle.IsNull || (handle.GraphId == graphId && handle.Value > 0 &&
            handle.Value <= Atomic.Add(ref state[0], 0)))
            return true;
        Fail();
        return false;
    }

    /// <summary>Validates a required non-null object and records invalid access.</summary>
    /// <param name="handle">The source object reference.</param>
    /// <returns>True when the reference is a valid non-null target.</returns>
    private bool ValidateObject(GpuObjectHandle handle)
    {
        if (!handle.IsNull && ValidateTarget(handle))
            return true;
        Fail();
        return false;
    }

    /// <summary>Validates an outgoing-reference slot before indexing device memory.</summary>
    /// <param name="handle">The source object.</param>
    /// <param name="index">The outgoing-reference index.</param>
    /// <returns>True when the object and slot are valid.</returns>
    private bool ValidateSlot(GpuObjectHandle handle, int index)
    {
        if (!ValidateObject(handle))
            return false;
        if (index >= 0 && index < lengths[handle.Index])
            return true;
        Fail();
        return false;
    }

    /// <summary>Records invalid graph access for host-side rejection.</summary>
    private void Fail() => Atomic.Exchange(ref state[3], 1);
}
