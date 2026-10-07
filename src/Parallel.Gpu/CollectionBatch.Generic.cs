using ILGPU;

namespace ParallelGpu;

/// <summary>Packs leaf collections of one numeric type into a contiguous device buffer.</summary>
/// <typeparam name="T">The numeric element type.</typeparam>
internal sealed class CollectionBatch<T> : CollectionBatch where T : unmanaged
{
    /// <summary>Leaf collections in packed-buffer order, allowing adjacent written slices to be downloaded together.</summary>
    private readonly List<CapturedCollection<T>> collections = new();
    /// <summary>The shared numeric device buffer created after all leaf slices are registered.</summary>
    private GpuBuffer<T>? buffer;
    /// <summary>The total reserved element count for this batch.</summary>
    private int length;

    /// <summary>Gets the shared device view after the batch has been uploaded.</summary>
    internal ArrayView<T> View => buffer!.View;

    /// <summary>Reserves a contiguous slice for a leaf collection.</summary>
    /// <param name="collection">The leaf collection to append.</param>
    /// <returns>The element offset of the reserved slice.</returns>
    /// <exception cref="OverflowException">The packed element count exceeds Int32.MaxValue.</exception>
    internal int Add(CapturedCollection<T> collection)
    {
        int offset = length;
        length = checked(length + collection.Snapshot.Length);
        collections.Add(collection);
        return offset;
    }

    /// <inheritdoc/>
    internal override void Upload(Gpu gpu)
    {
        var packed = new T[length];
        foreach (var collection in collections)
            collection.Snapshot.CopyTo(packed, collection.Offset);
        buffer = gpu.Upload(packed);
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    public override void Dispose() => buffer?.Dispose();
}
