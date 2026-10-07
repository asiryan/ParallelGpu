using System.Collections.Concurrent;
using System.Reflection;
using ILGPU;

namespace ParallelGpu;

/// <summary>Owns collection registration and device transfers for one compiled-loop invocation.</summary>
/// <param name="gpu">The executor that owns all device allocations.</param>
/// <remarks>Reference identity preserves aliases. Collections share one data buffer per numeric type
/// and common buffers for row metadata and write flags.</remarks>
internal sealed class CollectionTransfers(GpuParallel gpu) : IDisposable
{
    /// <summary>Typed registration delegates shared across invocations to avoid repeated generic reflection.</summary>
    private static readonly ConcurrentDictionary<Type, Func<CollectionTransfers, object, CapturedCollection>>
        factories = new();
    /// <summary>Host collection wrappers keyed by reference identity so repeated captures share one device view.</summary>
    private readonly Dictionary<object, CapturedCollection> collections =
        new(ReferenceIdentityComparer.Instance);
    /// <summary>Packed numeric buffers grouped by element type for the current invocation.</summary>
    private readonly Dictionary<Type, CollectionBatch> batches = new();
    /// <summary>Row descriptors stored as consecutive offset, length, and write-flag index triples.</summary>
    private readonly List<int> rows = new();
    /// <summary>The shared device row table, allocated only when row metadata is present.</summary>
    private GpuBuffer<int>? rowBuffer;
    /// <summary>One device write flag per unique leaf collection, used to select host copy-back.</summary>
    private GpuBuffer<int>? writeBuffer;
    /// <summary>The number of unique leaf collections and reserved write flags.</summary>
    private int collectionCount;

    /// <summary>Gets the packed row table containing element offsets, lengths, and write-flag indices.</summary>
    internal ArrayView<int> Rows => rowBuffer?.View ?? default;
    /// <summary>Gets the write flags used to select collections for host copy-back.</summary>
    internal ArrayView<int> Writes => writeBuffer?.View ?? default;

    /// <summary>Registers a host collection using its runtime numeric element type.</summary>
    /// <param name="source">The host collection to register.</param>
    /// <param name="elementType">The supported numeric element type.</param>
    /// <returns>The shared wrapper for the source object's identity.</returns>
    internal CapturedCollection Add(object source, Type elementType) =>
        factories.GetOrAdd(elementType, CreateFactory)(this, source);

    /// <summary>Builds a typed registration delegate without reflection on individual jagged rows.</summary>
    /// <param name="elementType">The numeric element type to bind.</param>
    /// <returns>A registration delegate cached for that element type.</returns>
    private static Func<CollectionTransfers, object, CapturedCollection> CreateFactory(Type elementType) =>
        (Func<CollectionTransfers, object, CapturedCollection>)typeof(CollectionTransfers)
            .GetMethod(nameof(AddNumeric), BindingFlags.NonPublic | BindingFlags.Static)!
            .MakeGenericMethod(elementType)
            .CreateDelegate(typeof(Func<CollectionTransfers, object, CapturedCollection>));

    /// <summary>Adapts typed registration to the cached untyped factory signature.</summary>
    /// <typeparam name="T">The numeric element type.</typeparam>
    /// <param name="transfers">The current invocation's registry.</param>
    /// <param name="source">The host collection to register.</param>
    /// <returns>The registered collection wrapper.</returns>
    private static CapturedCollection AddNumeric<T>(CollectionTransfers transfers, object source)
        where T : unmanaged => transfers.Add<T>(source);

    /// <summary>Registers a typed host collection and reuses wrappers for existing aliases.</summary>
    /// <typeparam name="T">The numeric element type.</typeparam>
    /// <param name="source">The host collection to register.</param>
    /// <returns>The shared wrapper for the source object's identity.</returns>
    internal CapturedCollection Add<T>(object source) where T : unmanaged
    {
        // Jagged rows and independently captured arrays must resolve to the same leaf slice.
        if (collections.TryGetValue(source, out var existing))
            return existing;
        CapturedCollection collection = source is T[][] jagged
            ? new CapturedJaggedCollection<T>(this, jagged)
            : new CapturedCollection<T>(this, source);
        collections.Add(source, collection);
        return collection;
    }

    /// <summary>Gets or creates the packed data batch for one element type.</summary>
    /// <typeparam name="T">The numeric element type.</typeparam>
    /// <returns>The invocation's batch for that type.</returns>
    internal CollectionBatch<T> Batch<T>() where T : unmanaged
    {
        if (!batches.TryGetValue(typeof(T), out var batch))
        {
            batch = new CollectionBatch<T>();
            batches.Add(typeof(T), batch);
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
        }
        return offset;
    }

    /// <summary>Allocates write flags and uploads all registered row metadata and numeric data.</summary>
    /// <remarks>Complete collection registration before calling this method.</remarks>
    internal void Upload()
    {
        // Zero-filled flags leave read-only captures on the device; writes select the slices to download.
        writeBuffer = gpu.Allocate<int>(collectionCount);
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
    }

    /// <summary>Releases all packed data and metadata buffers owned by this invocation.</summary>
    public void Dispose()
    {
        foreach (var batch in batches.Values)
            batch.Dispose();
        rowBuffer?.Dispose();
        writeBuffer?.Dispose();
    }
}
