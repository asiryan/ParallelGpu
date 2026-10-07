using System.Collections.Concurrent;
using System.Reflection;
using ILGPU;

namespace ParallelGpu;

// One invocation owns one data buffer per numeric type, plus shared row and write metadata.
internal sealed class CollectionTransfers(GpuParallel gpu) : IDisposable
{
    private static readonly ConcurrentDictionary<Type, Func<CollectionTransfers, object, CapturedCollection>>
        factories = new();
    private readonly Dictionary<object, CapturedCollection> collections =
        new(ReferenceIdentityComparer.Instance);
    private readonly Dictionary<Type, CollectionBatch> batches = new();
    private readonly List<int> rows = new();
    private GpuBuffer<int>? rowBuffer;
    private GpuBuffer<int>? writeBuffer;
    private int collectionCount;

    internal ArrayView<int> Rows => rowBuffer?.View ?? default;
    internal ArrayView<int> Writes => writeBuffer?.View ?? default;

    internal CapturedCollection Add(object source, Type elementType) =>
        factories.GetOrAdd(elementType, CreateFactory)(this, source);

    private static Func<CollectionTransfers, object, CapturedCollection> CreateFactory(Type elementType) =>
        (Func<CollectionTransfers, object, CapturedCollection>)typeof(CollectionTransfers)
            .GetMethod(nameof(AddNumeric), BindingFlags.NonPublic | BindingFlags.Static)!
            .MakeGenericMethod(elementType)
            .CreateDelegate(typeof(Func<CollectionTransfers, object, CapturedCollection>));

    private static CapturedCollection AddNumeric<T>(CollectionTransfers transfers, object source)
        where T : unmanaged => transfers.Add<T>(source);

    internal CapturedCollection Add<T>(object source) where T : unmanaged
    {
        if (collections.TryGetValue(source, out var existing))
            return existing;
        CapturedCollection collection = source is T[][] jagged
            ? new CapturedJaggedCollection<T>(this, jagged)
            : new CapturedCollection<T>(this, source);
        collections.Add(source, collection);
        return collection;
    }

    internal CollectionBatch<T> Batch<T>() where T : unmanaged
    {
        if (!batches.TryGetValue(typeof(T), out var batch))
        {
            batch = new CollectionBatch<T>();
            batches.Add(typeof(T), batch);
        }
        return (CollectionBatch<T>)batch;
    }

    internal int NextCollection() => collectionCount++;

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

    internal void Upload()
    {
        writeBuffer = gpu.Allocate<int>(collectionCount);
        if (rows.Count > 0)
            rowBuffer = gpu.Upload(rows.ToArray());
        foreach (var batch in batches.Values)
            batch.Upload(gpu);
    }

    internal void CopyBack()
    {
        int[] written = writeBuffer!.ToArray();
        foreach (var batch in batches.Values)
            batch.CopyBack(written);
    }

    public void Dispose()
    {
        foreach (var batch in batches.Values)
            batch.Dispose();
        rowBuffer?.Dispose();
        writeBuffer?.Dispose();
    }
}

internal abstract class CollectionBatch : IDisposable
{
    internal abstract void Upload(GpuParallel gpu);
    internal abstract void CopyBack(int[] written);
    public abstract void Dispose();
}

internal sealed class CollectionBatch<T> : CollectionBatch where T : unmanaged
{
    private readonly List<CapturedCollection<T>> collections = new();
    private GpuBuffer<T>? buffer;
    private int length;

    internal ArrayView<T> View => buffer!.View;

    internal int Add(CapturedCollection<T> collection)
    {
        int offset = length;
        length = checked(length + collection.Snapshot.Length);
        collections.Add(collection);
        return offset;
    }

    internal override void Upload(GpuParallel gpu)
    {
        var packed = new T[length];
        foreach (var collection in collections)
            collection.Snapshot.CopyTo(packed, collection.Offset);
        buffer = gpu.Upload(packed);
    }

    internal override void CopyBack(int[] written)
    {
        for (int first = 0; first < collections.Count;)
        {
            var collection = collections[first];
            if (written[collection.DirtyIndex] == 0 || collection.Snapshot.Length == 0)
            {
                first++;
                continue;
            }

            // Adjacent written collections share one device-to-host transfer.
            int end = first + 1;
            int limit = collection.Offset + collection.Snapshot.Length;
            while (end < collections.Count)
            {
                var next = collections[end];
                if (next.Snapshot.Length > 0 && written[next.DirtyIndex] == 0)
                    break;
                limit = next.Offset + next.Snapshot.Length;
                end++;
            }
            var result = new T[limit - collection.Offset];
            buffer!.CopyTo(collection.Offset, result);
            for (int i = first; i < end; i++)
            {
                var target = collections[i];
                if (target.Snapshot.Length > 0)
                    target.Apply(result.AsSpan(target.Offset - collection.Offset, target.Snapshot.Length));
            }
            first = end;
        }
    }

    public override void Dispose() => buffer?.Dispose();
}
