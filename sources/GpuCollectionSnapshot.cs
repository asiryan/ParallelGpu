namespace ParallelGpu;

/// <summary>Describes contiguous host storage exposed by an indexed collection adapter.</summary>
/// <typeparam name="T">A supported GPU loop element type.</typeparam>
/// <remarks>
/// <para>Collections sharing storage must return the same backing array or a shared identity for staged
/// snapshots. Shared identities require the same element type and full buffer length; an array or List
/// identity denotes that exact native storage. Slices transfer the full buffer, while read-only restrictions
/// apply to each view.</para>
/// <para>After a successful loop, written storage is downloaded and committed at most once per identity.
/// A Memory view sharing a memory-manager identity is updated before the user commit callback runs.
/// Callback exceptions can leave partial host changes.</para>
/// </remarks>
public readonly struct GpuCollectionSnapshot<T>
{
    /// <summary>Creates a snapshot covering an entire backing array.</summary>
    /// <param name="buffer">The shared backing array or staged full-storage snapshot.</param>
    /// <param name="isReadOnly">Whether device writes through this collection are forbidden.</param>
    /// <param name="commit">An optional host callback receiving the entire updated buffer when shared storage was written.</param>
    /// <param name="identity">The shared host storage identity, defaulting to the backing array.</param>
    /// <exception cref="ArgumentNullException">The backing buffer is null.</exception>
    public GpuCollectionSnapshot(T[] buffer, bool isReadOnly = false, Action<T[]>? commit = null, object? identity = null)
        : this(buffer, 0, buffer?.Length ?? 0, isReadOnly, commit, identity) { }

    /// <summary>Creates a snapshot covering a slice of shared contiguous storage.</summary>
    /// <param name="buffer">The full shared backing array or staged snapshot.</param>
    /// <param name="offset">The zero-based first element of the slice.</param>
    /// <param name="count">The number of accessible elements.</param>
    /// <param name="isReadOnly">Whether device writes through this collection are forbidden.</param>
    /// <param name="commit">An optional host callback receiving the entire updated buffer when shared storage was written.</param>
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
