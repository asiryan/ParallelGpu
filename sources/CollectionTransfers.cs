using System.Collections.Concurrent;
using System.Reflection;
using ILGPU;

namespace ParallelGpu;

/// <summary>Owns collection registration and device transfers for one compiled-loop invocation.</summary>
/// <param name="gpu">The executor that owns all device allocations.</param>
/// <remarks>Reference identity preserves aliases. Collections share one data buffer per element type
/// and common buffers for row metadata and write flags.</remarks>
internal sealed class CollectionTransfers(Gpu gpu) : IDisposable
{
    /// <summary>Typed registration delegates shared across invocations to avoid repeated generic reflection.</summary>
    private static readonly ConcurrentDictionary<Type, Func<CollectionTransfers, object, Type, CapturedCollection>>
        factories = new();
    /// <summary>Host wrappers grouped by element type, capture declaration, and reference identity.</summary>
    private readonly Dictionary<(Type Element, Type? Declaration), Dictionary<object, CapturedCollection>> collections = new();
    /// <summary>Reference identities shared by all typed representations of the same host object.</summary>
    private readonly Dictionary<object, int> identities = new(ReferenceIdentityComparer.Instance);
    /// <summary>Shared staged snapshots keyed by their host storage identities.</summary>
    private readonly Dictionary<object, object> adapterSnapshots = new(ReferenceIdentityComparer.Instance);
    /// <summary>Native memory copyback and one user commit per storage identity, sharing the leaf write flag.</summary>
    private readonly Dictionary<object, (int DirtyIndex, Action? CopyMemory, Action? Commit)> adapterCommits =
        new(ReferenceIdentityComparer.Instance);
    /// <summary>Packed device buffers grouped by element type for the current invocation.</summary>
    private readonly Dictionary<Type, CollectionBatch> batches = new();
    /// <summary>Row descriptors packed with offsets, lengths, write flags, shape offsets, ranks, and identities.</summary>
    private readonly List<int> rows = new();
    /// <summary>Dimension triples for array shapes that cannot use the inline flat descriptor.</summary>
    private readonly List<int> shapes = new();
    /// <summary>The shared device dimension table.</summary>
    private GpuBuffer<int>? shapeBuffer;
    /// <summary>The next nonnull reference identity.</summary>
    private int nextIdentity = 1;
    /// <summary>Gets or sets the invocation context before uploading collection views.</summary>
    internal DeviceExecutionContext Context { get; set; }
    /// <summary>Gets the largest physical element size among registered batches.</summary>
    internal int MaximumElementSize { get; private set; } = 1;
    /// <summary>The shared device row table, allocated only when row metadata is present.</summary>
    private GpuBuffer<int>? rowBuffer;
    /// <summary>One device write flag per unique leaf collection, used to select host copy-back.</summary>
    private GpuBuffer<int>? writeBuffer;
    /// <summary>The number of unique leaf collections and reserved write flags.</summary>
    private int collectionCount;

    /// <summary>Gets the packed nested-array child descriptors.</summary>
    internal ArrayView<int> Rows => rowBuffer?.View ?? default;
    /// <summary>Gets the dimension table storing length, lower bound, and stride triples.</summary>
    internal ArrayView<int> Shapes => shapeBuffer?.View ?? default;
    /// <summary>Gets the write flags used to select collections for host copy-back.</summary>
    internal ArrayView<int> Writes => writeBuffer?.View ?? default;

    /// <summary>Registers a host collection using the supplied element type.</summary>
    /// <param name="source">The host collection to register.</param>
    /// <param name="elementType">The supported element type.</param>
    /// <param name="declaredType">The capture declaration selecting its native or adapted representation.</param>
    /// <returns>The wrapper shared by aliases using the same element type and capture representation.</returns>
    internal CapturedCollection Add(object? source, Type elementType, Type declaredType) =>
        source is null ? new CapturedNullCollection(this) : factories.GetOrAdd(elementType, CreateFactory)(this, source, declaredType);

    /// <summary>Builds a typed registration delegate without reflection on individual jagged rows.</summary>
    /// <param name="elementType">The supported element type to bind.</param>
    /// <returns>A registration delegate cached for that element type.</returns>
    private static Func<CollectionTransfers, object, Type, CapturedCollection> CreateFactory(Type elementType) =>
        (Func<CollectionTransfers, object, Type, CapturedCollection>)typeof(CollectionTransfers)
            .GetMethod(nameof(AddTyped), BindingFlags.NonPublic | BindingFlags.Static)!
            .MakeGenericMethod(elementType)
            .CreateDelegate(typeof(Func<CollectionTransfers, object, Type, CapturedCollection>));

    /// <summary>Adapts typed registration to the cached untyped factory signature.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="transfers">The current invocation's registry.</param>
    /// <param name="source">The host collection to register.</param>
    /// <param name="declaredType">The capture declaration selecting its representation.</param>
    /// <returns>The registered collection wrapper.</returns>
    private static CapturedCollection AddTyped<T>(CollectionTransfers transfers, object source, Type declaredType)
        => transfers.Add<T>(source, declaredType);

    /// <summary>Registers a typed host collection and reuses wrappers for existing aliases.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="source">The host collection to register.</param>
    /// <param name="declaredType">The optional declaration; internal backing registrations use native storage.</param>
    /// <returns>The wrapper shared by aliases using the same element type and capture representation.</returns>
    internal CapturedCollection Add<T>(object source, Type? declaredType = null)
    {
        bool native = (source is Array || source is List<T>) &&
            (declaredType is null || CapturedCollection.IsNativeCollection(declaredType));
        var key = (typeof(T), native ? null : declaredType ?? source.GetType());
        if (!collections.TryGetValue(key, out var typedCollections))
        {
            typedCollections = new(ReferenceIdentityComparer.Instance);
            collections.Add(key, typedCollections);
        }
        // An object may expose several registered indexers with independent element types.
        if (typedCollections.TryGetValue(source, out var existing))
            return existing;
        CapturedCollection collection;
        if (!native &&
            GpuCollectionAdapters.TryCapture<T>(source, declaredType, adapterSnapshots, out var snapshot, out var copyMemory))
        {
            var backing = (CapturedCollection<T>)Add<T>(snapshot.Identity is Array || snapshot.Identity is List<T>
                ? snapshot.Identity : snapshot.Buffer);
            if (snapshot.Identity is Array || snapshot.Identity is List<T>)
            {
                // Native storage and its staged adapters must share one leaf regardless of capture order.
                snapshot = new(backing.Snapshot, snapshot.Offset, snapshot.Count, snapshot.IsReadOnly,
                    snapshot.Commit, snapshot.Identity);
                adapterSnapshots[snapshot.Identity] = snapshot;
            }
            collection = new CapturedSliceCollection<T>(this, backing, snapshot,
                arraySegment: source is ArraySegment<T>,
                nullBackedSegment: source is ArraySegment<T> segment && segment.Array is null,
                nativeList: source is List<T> && ReferenceEquals(snapshot.Identity, source));
            if (copyMemory is not null || snapshot.Commit is not null)
            {
                if (!adapterCommits.TryGetValue(snapshot.Identity, out var callbacks))
                    callbacks = (backing.DirtyIndex, null, null);
                // Native memory may be captured before or after a notification-only user adapter.
                // Keep both actions; a later capture must not replace the first user callback.
                if (callbacks.CopyMemory is null && copyMemory is not null)
                    callbacks.CopyMemory = () => copyMemory(snapshot.Buffer);
                if (callbacks.Commit is null && snapshot.Commit is not null)
                    callbacks.Commit = () => snapshot.Commit(snapshot.Buffer);
                adapterCommits[snapshot.Identity] = callbacks;
            }
        }
        else
            collection = source is Array nested && nested.GetType().GetElementType()!.IsArray
                ? new CapturedJaggedCollection<T>(this, nested)
                : new CapturedCollection<T>(this, source);
        if (!identities.TryGetValue(source, out int identity))
        {
            identity = checked(nextIdentity++);
            identities.Add(source, identity);
        }
        collection.Identity = identity;
        typedCollections.Add(source, collection);
        return collection;
    }

    /// <summary>Gets or creates the packed data batch for one element type.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <returns>The invocation's batch for that type.</returns>
    internal CollectionBatch<T> Batch<T>()
    {
        if (!batches.TryGetValue(typeof(T), out var batch))
        {
            batch = new CollectionBatch<T>();
            batches.Add(typeof(T), batch);
            MaximumElementSize = Math.Max(MaximumElementSize, ((CollectionBatch<T>)batch).ElementSize);
        }
        return (CollectionBatch<T>)batch;
    }

    /// <summary>Reserves a unique write-flag index for a leaf collection.</summary>
    /// <returns>The zero-based index of the reserved flag.</returns>
    internal int NextCollection() => collectionCount++;

    /// <summary>Appends row descriptors to the shared packed row table.</summary>
    /// <param name="descriptors">The child descriptors to append.</param>
    /// <returns>The integer offset of the first appended descriptor in the table.</returns>
    internal int AddRows(RowDescriptor[] descriptors)
    {
        int offset = rows.Count;
        foreach (var row in descriptors)
        {
            rows.Add(row.Offset);
            rows.Add(row.Length);
            rows.Add(row.DirtyIndex);
            rows.Add(row.ShapeOffset);
            rows.Add(row.Rank);
            rows.Add(row.Identity);
        }
        return offset;
    }

    /// <summary>Registers an array's row-major shape while preserving its native lower bounds.</summary>
    /// <param name="array">The host array.</param>
    /// <returns>The first dimension's integer offset, or minus one for an inline zero-based flat shape.</returns>
    internal int AddShape(Array array)
    {
        if (array.Rank == 1 && array.GetLowerBound(0) == 0)
            return -1;
        int offset = shapes.Count;
        var metadata = new int[checked(array.Rank * 3)];
        int stride = 1;
        for (int dimension = array.Rank - 1; dimension >= 0; dimension--)
        {
            int length = array.GetLength(dimension);
            int position = dimension * 3;
            metadata[position] = length;
            metadata[position + 1] = array.GetLowerBound(dimension);
            metadata[position + 2] = stride;
            // Empty arrays can have theoretical strides exceeding Int32; no valid access can use them.
            stride = array.Length == 0 && length != 0 && stride > int.MaxValue / length
                ? 0 : checked(stride * length);
        }
        shapes.AddRange(metadata);
        return offset;
    }

    /// <summary>Uses the descriptor's inline length for a zero-based one-dimensional shape.</summary>
    /// <param name="length">The collection length.</param>
    /// <returns>Minus one, selecting the descriptor's inline shape.</returns>
    internal int AddFlatShape(int length) => -1;

    /// <summary>Allocates write flags and uploads all registered row metadata and element data.</summary>
    /// <remarks>Complete collection registration before calling this method.</remarks>
    internal void Upload()
    {
        // Only leaves marked as written, including writable ref accesses, need copyback.
        if (collectionCount > 0)
            writeBuffer = gpu.Allocate<int>(collectionCount);
        if (shapes.Count > 0)
            shapeBuffer = gpu.Upload(shapes.ToArray());
        if (rows.Count > 0)
            rowBuffer = gpu.Upload(rows.ToArray());
        foreach (var batch in batches.Values)
            batch.Upload(gpu);
    }

    /// <summary>Reads write flags, copies written backing leaves to the host, and commits staged adapters.</summary>
    /// <remarks>All launches must succeed and deferred errors must be validated before calling this.
    /// Each written adapter storage identity is committed at most once.</remarks>
    internal void CopyBack()
    {
        if (writeBuffer is null)
            return;
        int[] written = writeBuffer.ToArray();
        foreach (var batch in batches.Values)
            batch.CopyBack(written);
        foreach (var commit in adapterCommits.Values)
            if (written[commit.DirtyIndex] != 0)
            {
                commit.CopyMemory?.Invoke();
                commit.Commit?.Invoke();
            }
    }

    /// <summary>Releases all packed data and metadata buffers owned by this invocation.</summary>
    public void Dispose()
    {
        foreach (var batch in batches.Values)
            batch.Dispose();
        rowBuffer?.Dispose();
        shapeBuffer?.Dispose();
        writeBuffer?.Dispose();
    }
}
