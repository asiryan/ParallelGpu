namespace ParallelGpu;

/// <summary>Describes contiguous host storage exposed by an indexed collection adapter.</summary>
/// <typeparam name="T">The runtime-validated GPU loop element type.</typeparam>
/// <remarks>Return the same backing array for shared storage. For staged storage, supply a shared identity
/// and a commit callback; one backing buffer and at most one commit are used per identity and invocation.
/// Shared identities require the same element type and full buffer length. An array or List identity
/// denotes that exact native storage.
/// Slices transfer the full backing buffer; read-only restrictions apply to each view. Commits run after
/// successful execution and downloads, only for backing storage marked as written. Callback exceptions
/// can leave partial host changes.</remarks>
public readonly struct GpuCollectionSnapshot<T>
{
    /// <summary>Creates a snapshot covering an entire backing array.</summary>
    /// <param name="buffer">The shared backing array or staged full-storage snapshot.</param>
    /// <param name="isReadOnly">Whether device writes through this collection are forbidden.</param>
    /// <param name="commit">An optional callback receiving the updated entire buffer after a successful loop that marked the shared storage as written.</param>
    /// <param name="identity">The shared host storage identity, defaulting to the backing array.</param>
    /// <exception cref="ArgumentNullException">The backing buffer is null.</exception>
    public GpuCollectionSnapshot(T[] buffer, bool isReadOnly = false, Action<T[]>? commit = null, object? identity = null)
        : this(buffer, 0, buffer?.Length ?? 0, isReadOnly, commit, identity) { }

    /// <summary>Creates a snapshot covering a slice of shared contiguous storage.</summary>
    /// <param name="buffer">The full shared backing array or staged snapshot.</param>
    /// <param name="offset">The zero-based first element of the slice.</param>
    /// <param name="count">The number of accessible elements.</param>
    /// <param name="isReadOnly">Whether device writes through this collection are forbidden.</param>
    /// <param name="commit">An optional callback receiving the updated entire buffer after a successful loop that marked the shared storage as written.</param>
    /// <param name="identity">The shared host storage identity, defaulting to the backing array.</param>
    /// <exception cref="ArgumentNullException">The backing buffer is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The offset or count is outside the backing buffer.</exception>
    public GpuCollectionSnapshot(T[] buffer, int offset, int count, bool isReadOnly = false,
        Action<T[]>? commit = null, object? identity = null)
    {
        Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        if (offset < 0 || offset > buffer.Length)
            throw new ArgumentOutOfRangeException(nameof(offset));
        if (count < 0 || count > buffer.Length - offset)
            throw new ArgumentOutOfRangeException(nameof(count));
        Offset = offset;
        Count = count;
        IsReadOnly = isReadOnly;
        Commit = commit;
        Identity = identity ?? buffer;
    }

    /// <summary>Gets the full contiguous storage shared by related collection slices.</summary>
    public T[] Buffer { get; }
    /// <summary>Gets the slice's starting element offset.</summary>
    public int Offset { get; }
    /// <summary>Gets the slice's accessible element count.</summary>
    public int Count { get; }
    /// <summary>Gets whether writes through this view are forbidden.</summary>
    public bool IsReadOnly { get; }
    /// <summary>Gets the optional host callback used after successful execution and download of written storage.</summary>
    public Action<T[]>? Commit { get; }
    /// <summary>Gets the host storage identity used to unify staged aliases.</summary>
    public object Identity { get; }
}
