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
    private static readonly ConcurrentDictionary<Type, Func<CollectionTransfers, object, CapturedCollection>>
        factories = new();
    /// <summary>Host collection wrappers keyed by reference identity so repeated captures share one device view.</summary>
    private readonly Dictionary<object, CapturedCollection> collections =
        new(ReferenceIdentityComparer.Instance);
    /// <summary>Shared staged snapshots keyed by their host storage identities.</summary>
    private readonly Dictionary<object, object> adapterSnapshots = new(ReferenceIdentityComparer.Instance);
    /// <summary>One host commit per storage identity, retaining its shared backing leaf write flag.</summary>
    private readonly Dictionary<object, (int DirtyIndex, Action Commit)> adapterCommits =
        new(ReferenceIdentityComparer.Instance);
    /// <summary>Packed device buffers grouped by element type for the current invocation.</summary>
    private readonly Dictionary<Type, CollectionBatch> batches = new();
    /// <summary>Row descriptors packed with offsets, lengths, write flags, shape offsets, ranks, and identities.</summary>
    private readonly List<int> rows = new();
    /// <summary>Dimension triples shared by every registered array and list.</summary>
    private readonly List<int> shapes = new();
    /// <summary>The shared device dimension table.</summary>
    private GpuBuffer<int>? shapeBuffer;
    /// <summary>The next nonnull reference identity.</summary>
    private int nextIdentity = 1;
    /// <summary>Gets or sets the invocation context before uploading collection views.</summary>
    internal DeviceExecutionContext Context { get; set; }
    /// <summary>Gets the largest physical element size for safe per-thread error references.</summary>
    internal int MaximumElementSize { get; private set; } = 1;
    /// <summary>The shared device row table, allocated only when row metadata is present.</summary>
    private GpuBuffer<int>? rowBuffer;
    /// <summary>One device write flag per unique leaf collection, used to select host copy-back.</summary>
    private GpuBuffer<int>? writeBuffer;
    /// <summary>The number of unique leaf collections and reserved write flags.</summary>
    private int collectionCount;

    /// <summary>Gets the packed row table containing element offsets, lengths, and write-flag indices.</summary>
    internal ArrayView<int> Rows => rowBuffer?.View ?? default;
    /// <summary>Gets the dimension table storing length, lower bound, and stride triples.</summary>
    internal ArrayView<int> Shapes => shapeBuffer?.View ?? default;
    /// <summary>Gets the write flags used to select collections for host copy-back.</summary>
    internal ArrayView<int> Writes => writeBuffer?.View ?? default;

    /// <summary>Registers a host collection using its runtime supported element type.</summary>
    /// <param name="source">The host collection to register.</param>
    /// <param name="elementType">The supported element type.</param>
    /// <returns>The shared wrapper for the source object's identity.</returns>
    internal CapturedCollection Add(object? source, Type elementType) =>
        source is null ? new CapturedNullCollection(this) : factories.GetOrAdd(elementType, CreateFactory)(this, source);

    /// <summary>Builds a typed registration delegate without reflection on individual jagged rows.</summary>
    /// <param name="elementType">The supported element type to bind.</param>
    /// <returns>A registration delegate cached for that element type.</returns>
    private static Func<CollectionTransfers, object, CapturedCollection> CreateFactory(Type elementType) =>
        (Func<CollectionTransfers, object, CapturedCollection>)typeof(CollectionTransfers)
            .GetMethod(nameof(AddTyped), BindingFlags.NonPublic | BindingFlags.Static)!
            .MakeGenericMethod(elementType)
            .CreateDelegate(typeof(Func<CollectionTransfers, object, CapturedCollection>));

    /// <summary>Adapts typed registration to the cached untyped factory signature.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="transfers">The current invocation's registry.</param>
    /// <param name="source">The host collection to register.</param>
    /// <returns>The registered collection wrapper.</returns>
    private static CapturedCollection AddTyped<T>(CollectionTransfers transfers, object source)
        => transfers.Add<T>(source);

    /// <summary>Registers a typed host collection and reuses wrappers for existing aliases.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="source">The host collection to register.</param>
    /// <returns>The shared wrapper for the source object's identity.</returns>
    internal CapturedCollection Add<T>(object source)
    {
        // Jagged rows and independently captured arrays must resolve to the same leaf slice.
        if (collections.TryGetValue(source, out var existing))
            return existing;
        CapturedCollection collection;
        if (source is not Array && source is not List<T> &&
            GpuCollectionAdapters.TryCapture<T>(source, adapterSnapshots, out var snapshot))
        {
            var backing = (CapturedCollection<T>)Add<T>(snapshot.Buffer);
            collection = new CapturedSliceCollection<T>(this, backing, snapshot,
                arraySegment: source is ArraySegment<T>,
                nullBackedSegment: source is ArraySegment<T> segment && segment.Array is null);
            if (snapshot.Commit is not null && !adapterCommits.ContainsKey(snapshot.Identity))
                adapterCommits.Add(snapshot.Identity, (backing.DirtyIndex, () => snapshot.Commit(snapshot.Buffer)));
        }
        else
            collection = source is Array nested && nested.GetType().GetElementType()!.IsArray
                ? new CapturedJaggedCollection<T>(this, nested)
                : new CapturedCollection<T>(this, source);
        collection.Identity = checked(nextIdentity++);
        collections.Add(source, collection);
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
    /// <param name="descriptors">The row offsets, lengths, and write-flag indices to append.</param>
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

    /// <summary>Appends the row-major shape of an array, retaining its native lower bounds.</summary>
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
        // Zero-filled flags leave read-only captures on the device; writes select the slices to download.
        writeBuffer = gpu.Allocate<int>(collectionCount);
        if (shapes.Count > 0)
            shapeBuffer = gpu.Upload(shapes.ToArray());
        if (rows.Count > 0)
            rowBuffer = gpu.Upload(rows.ToArray());
        foreach (var batch in batches.Values)
            batch.Upload(gpu);
    }

    /// <summary>Reads write flags and copies affected collection slices into their original host objects.</summary>
    /// <remarks>Device execution must have completed before this method is called.</remarks>
    internal void CopyBack()
    {
        int[] written = writeBuffer!.ToArray();
        foreach (var batch in batches.Values)
            batch.CopyBack(written);
        foreach (var commit in adapterCommits.Values)
            if (written[commit.DirtyIndex] != 0)
                commit.Commit();
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
