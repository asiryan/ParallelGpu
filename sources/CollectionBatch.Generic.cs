using ILGPU;

namespace ParallelGpu;

/// <summary>Packs leaf collections of one supported type into a contiguous device buffer.</summary>
/// <typeparam name="T">The host element type.</typeparam>
internal sealed class CollectionBatch<T> : CollectionBatch
{
    /// <summary>The minimum average slice size for avoiding host packing without multiplying tiny transfers.</summary>
    private const int MinimumDirectTransferBytes = 1024 * 1024;
    /// <summary>Leaf collections in packed-buffer order, allowing adjacent written slices to be downloaded together.</summary>
    private readonly List<CapturedCollection<T>> collections = new();
    /// <summary>The shared 32-bit storage used for Boolean and character elements.</summary>
    private GpuBuffer<int>? paddedBuffer;
    /// <summary>Runtime-bound native or mirrored storage; host nullable types need no unmanaged generic constraint.</summary>
    private readonly DeviceValueBuffer<T>? valueBuffer = UsesPaddedStorage ? null : DeviceValueBuffer<T>.Create();
    /// <summary>The total reserved element count for this batch.</summary>
    private int length;

    /// <summary>Gets the shared byte view after the batch has been uploaded.</summary>
    internal ArrayView<byte> View => UsesPaddedStorage
        ? paddedBuffer!.View.Cast<byte>() : valueBuffer!.View;

    /// <summary>Gets the physical distance between elements in bytes, including Boolean and character padding.</summary>
    internal int ElementSize => UsesPaddedStorage ? sizeof(int) : valueBuffer!.ElementSize;

    /// <summary>Gets whether this batch uses aligned 32-bit slots for its host element type.</summary>
    private static bool UsesPaddedStorage => typeof(T) == typeof(bool) || typeof(T) == typeof(char);

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
        if (UsesPaddedStorage)
        {
            // Aligned slots avoid relying on ILGPU Boolean and character layouts matching their CLR layouts.
            var padded = new int[length];
            foreach (var collection in collections)
                Pack(collection.Snapshot, padded.AsSpan(collection.Offset, collection.Snapshot.Length));
            paddedBuffer = gpu.Upload(padded);
            return;
        }
        // A single snapshot is already contiguous; upload it without an extra host staging array.
        if (collections.Count == 1)
        {
            valueBuffer!.Upload(gpu, collections[0].Snapshot);
            return;
        }
        // Direct slice uploads avoid a second full host buffer for large batches.
        // Pack small batches to share one transfer.
        if ((long)length * ElementSize >= (long)collections.Count * MinimumDirectTransferBytes)
        {
            valueBuffer!.Allocate(gpu, length);
            foreach (var collection in collections)
                valueBuffer!.CopyFrom(collection.Offset, collection.Snapshot);
            return;
        }
        var packed = new T[length];
        foreach (var collection in collections)
            collection.Snapshot.CopyTo(packed, collection.Offset);
        valueBuffer!.Upload(gpu, packed);
    }

    /// <summary>Converts Boolean or UTF-16 character values into aligned device slots.</summary>
    /// <param name="source">The host elements to convert.</param>
    /// <param name="destination">The matching device slots.</param>
    private static void Pack(T[] source, Span<int> destination)
    {
        if (typeof(T) == typeof(bool))
        {
            var values = (bool[])(object)source;
            for (int i = 0; i < values.Length; i++)
                destination[i] = values[i] ? 1 : 0;
        }
        else
        {
            var values = (char[])(object)source;
            for (int i = 0; i < values.Length; i++)
                destination[i] = values[i];
        }
    }

    /// <summary>Downloads a slice and converts padded slots back to their host element representation.</summary>
    /// <param name="offset">The starting element offset in the packed buffer.</param>
    /// <param name="destination">The host array to receive the slice.</param>
    private void Download(int offset, T[] destination)
    {
        if (!UsesPaddedStorage)
        {
            valueBuffer!.Download(offset, destination);
            return;
        }
        var padded = new int[destination.Length];
        paddedBuffer!.CopyTo(offset, padded);
        if (typeof(T) == typeof(bool))
        {
            var values = (bool[])(object)destination;
            for (int i = 0; i < values.Length; i++)
                values[i] = padded[i] != 0;
        }
        else
        {
            var values = (char[])(object)destination;
            for (int i = 0; i < values.Length; i++)
                values[i] = unchecked((char)padded[i]);
        }
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
            if (end == first + 1)
            {
                // Flat-array snapshots already point to the destination; other collections
                // reuse their snapshot instead of allocating another full-sized host array.
                Download(collection.Offset, collection.Snapshot);
                collection.Apply(collection.Snapshot);
                first = end;
                continue;
            }
            var result = new T[limit - collection.Offset];
            Download(collection.Offset, result);
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
    public override void Dispose()
    {
        paddedBuffer?.Dispose();
        valueBuffer?.Dispose();
    }
}
