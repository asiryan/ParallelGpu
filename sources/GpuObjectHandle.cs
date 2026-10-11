namespace ParallelGpu;

/// <summary>A nullable, graph-specific device reference to an object record.</summary>
/// <remarks>Handles remain stable while their graph is alive. The default value represents null.
/// Handles from different graphs cannot be mixed. Object allocation and reference manipulation
/// are explicit; a handle does not enable arbitrary managed class methods on a GPU.</remarks>
public readonly struct GpuObjectHandle : IEquatable<GpuObjectHandle>
{
    /// <summary>The owning graph's unique identity.</summary>
    internal readonly int GraphId;
    /// <summary>The one-based record index, or zero for null.</summary>
    internal readonly int Value;

    /// <summary>Builds a graph-specific reference from an internal record index.</summary>
    /// <param name="graphId">The owning graph identity.</param>
    /// <param name="value">The one-based record index.</param>
    internal GpuObjectHandle(int graphId, int value)
    {
        GraphId = graphId;
        Value = value;
    }

    /// <summary>Gets whether this handle represents null.</summary>
    public bool IsNull => Value == 0;

    /// <summary>Gets the zero-based record index, or -1 for null.</summary>
    public int Index => Value - 1;

    /// <summary>Compares graph identity and record identity; all null handles compare equal.</summary>
    /// <param name="other">The handle to compare.</param>
    /// <returns>True when both handles identify the same object or both are null.</returns>
    public bool Equals(GpuObjectHandle other) => Value == other.Value && (Value == 0 || GraphId == other.GraphId);

    /// <summary>Compares this handle with a boxed handle.</summary>
    /// <param name="obj">The boxed handle.</param>
    /// <returns>True when it identifies the same object or both handles are null.</returns>
    public override bool Equals(object? obj) => obj is GpuObjectHandle other && Equals(other);

    /// <summary>Returns a hash of graph and object identity.</summary>
    /// <returns>The identity hash, or zero for null.</returns>
    public override int GetHashCode() => IsNull ? 0 : unchecked(GraphId * 397 ^ Value);

    /// <summary>Tests whether two handles identify the same object.</summary>
    /// <param name="left">The first handle.</param>
    /// <param name="right">The second handle.</param>
    /// <returns>True when both references are equal.</returns>
    public static bool operator ==(GpuObjectHandle left, GpuObjectHandle right) => left.Equals(right);

    /// <summary>Tests whether two handles identify different objects.</summary>
    /// <param name="left">The first handle.</param>
    /// <param name="right">The second handle.</param>
    /// <returns>True when the references differ.</returns>
    public static bool operator !=(GpuObjectHandle left, GpuObjectHandle right) => !left.Equals(right);
}

/// <summary>Maps host objects into unmanaged records and ordered graph references.</summary>
/// <typeparam name="TObject">The managed host object type.</typeparam>
/// <typeparam name="TRecord">The unmanaged, ILGPU-supported device payload.</typeparam>
/// <remarks>All methods execute on the host. Encode class-specific values and type tags in the
/// record. The adapter determines which fields and references are represented and how new records
/// create host instances; arbitrary virtual methods and constructors are not executed on the GPU.</remarks>
public interface IGpuObjectGraphAdapter<TObject, TRecord> where TObject : class where TRecord : unmanaged
{
    /// <summary>Reads the fixed-size payload of a host object.</summary>
    /// <param name="source">The existing host object.</param>
    /// <returns>Its device-compatible payload.</returns>
    TRecord ReadValue(TObject source);

    /// <summary>Reads the ordered outgoing references of a host object.</summary>
    /// <param name="source">The existing host object.</param>
    /// <returns>The references, including null entries and shared or cyclic references.</returns>
    IReadOnlyList<TObject?> ReadReferences(TObject source);

    /// <summary>Creates an unlinked host object for a newly allocated device record.</summary>
    /// <param name="value">The new record's payload.</param>
    /// <returns>A distinct non-null host instance.</returns>
    TObject CreateObject(TRecord value);

    /// <summary>Applies a downloaded payload to an existing or newly created host object.</summary>
    /// <param name="target">The stable host instance.</param>
    /// <param name="value">The downloaded device payload.</param>
    void WriteValue(TObject target, TRecord value);

    /// <summary>Applies downloaded outgoing references after all host objects exist.</summary>
    /// <param name="target">The stable host instance.</param>
    /// <param name="references">The ordered references with host identity and null values preserved.</param>
    void WriteReferences(TObject target, IReadOnlyList<TObject?> references);
}

/// <summary>An immutable downloaded snapshot of a device object graph.</summary>
/// <typeparam name="TRecord">The unmanaged record type.</typeparam>
/// <remarks>The snapshot owns host copies and remains usable after its device graph is disposed.</remarks>
public sealed class GpuObjectGraphSnapshot<TRecord> where TRecord : unmanaged
{
    /// <summary>The owning graph's identity, used to validate handles.</summary>
    private readonly int graphId;
    /// <summary>The downloaded record payloads.</summary>
    internal readonly TRecord[] Data;
    /// <summary>The downloaded outgoing reference lists.</summary>
    internal readonly GpuObjectHandle[][] Links;

    /// <summary>Wraps validated host copies in read-only collections.</summary>
    /// <param name="graphId">The source graph identity.</param>
    /// <param name="values">The downloaded payloads.</param>
    /// <param name="references">The downloaded outgoing references.</param>
    /// <param name="roots">The downloaded root references.</param>
    internal GpuObjectGraphSnapshot(int graphId, TRecord[] values, GpuObjectHandle[][] references, GpuObjectHandle[] roots)
    {
        this.graphId = graphId;
        Data = values;
        Links = references;
        Values = Array.AsReadOnly(values);
        References = Array.AsReadOnly<IReadOnlyList<GpuObjectHandle>>(
            references.Select(array => (IReadOnlyList<GpuObjectHandle>)Array.AsReadOnly(array)).ToArray());
        Roots = Array.AsReadOnly(roots);
    }

    /// <summary>Gets one immutable value per non-null graph object.</summary>
    public IReadOnlyList<TRecord> Values { get; }

    /// <summary>Gets the ordered outgoing references of each object.</summary>
    public IReadOnlyList<IReadOnlyList<GpuObjectHandle>> References { get; }

    /// <summary>Gets the downloaded graph roots, including null entries.</summary>
    public IReadOnlyList<GpuObjectHandle> Roots { get; }

    /// <summary>Reads a payload using a handle belonging to this snapshot's graph.</summary>
    /// <param name="handle">A non-null graph reference.</param>
    /// <returns>The copied payload.</returns>
    /// <exception cref="ArgumentException">The handle is null, foreign, or unavailable in this snapshot.</exception>
    public TRecord Read(GpuObjectHandle handle)
    {
        Validate(handle);
        return Data[handle.Index];
    }

    /// <summary>Reads the immutable outgoing references of a graph object.</summary>
    /// <param name="handle">A non-null graph reference.</param>
    /// <returns>The ordered outgoing references.</returns>
    /// <exception cref="ArgumentException">The handle is null, foreign, or unavailable in this snapshot.</exception>
    public IReadOnlyList<GpuObjectHandle> GetReferences(GpuObjectHandle handle)
    {
        Validate(handle);
        return References[handle.Index];
    }

    /// <summary>Rejects invalid graph references before host indexing.</summary>
    /// <param name="handle">The reference to validate.</param>
    private void Validate(GpuObjectHandle handle)
    {
        if (handle.IsNull || handle.GraphId != graphId || handle.Index < 0 || handle.Index >= Data.Length)
            throw new ArgumentException("The handle does not identify an object in this graph snapshot.", nameof(handle));
    }
}
