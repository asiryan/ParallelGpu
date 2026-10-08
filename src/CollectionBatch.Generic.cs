using System.Runtime.InteropServices;
using ILGPU;

namespace ParallelGpu;

/// <summary>Packs leaf collections of one supported type into a contiguous device buffer.</summary>
/// <typeparam name="T">The host element type.</typeparam>
internal sealed class CollectionBatch<T> : CollectionBatch where T : unmanaged
{
    /// <summary>Leaf collections in packed-buffer order, allowing adjacent written slices to be downloaded together.</summary>
    private readonly List<CapturedCollection<T>> collections = new();
    /// <summary>The shared device buffer for elements with a native storage representation.</summary>
    private GpuBuffer<T>? buffer;
    /// <summary>The shared 32-bit storage used for Boolean and character elements.</summary>
    private GpuBuffer<int>? paddedBuffer;
    /// <summary>Converted storage for structures needing a public blittable device mirror.</summary>
    private readonly DeviceValueBuffer<T>? mappedBuffer =
        !typeof(T).IsPrimitive && DeviceValueType.MapType(typeof(T)) != typeof(T)
            ? DeviceValueBuffer<T>.Create() : null;
    /// <summary>The total reserved element count for this batch.</summary>
    private int length;

    /// <summary>Gets the shared byte view after the batch has been uploaded.</summary>
    internal ArrayView<byte> View => UsesPaddedStorage
        ? paddedBuffer!.View.Cast<byte>() : mappedBuffer?.View ?? buffer!.View.Cast<byte>();

    /// <summary>Gets the physical distance between elements in bytes, including Boolean and character padding.</summary>
    internal int ElementSize => UsesPaddedStorage ? sizeof(int) : mappedBuffer?.ElementSize ?? Interop.SizeOf<T>();

    /// <summary>Gets whether the device requires aligned 32-bit slots for this host element type.</summary>
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
        var packed = new T[length];
        foreach (var collection in collections)
            collection.Snapshot.CopyTo(packed, collection.Offset);
        if (mappedBuffer is not null)
            mappedBuffer.Upload(gpu, packed);
        else
            buffer = gpu.Upload(packed);
    }

    /// <summary>Converts Boolean or UTF-16 character values into aligned device slots.</summary>
    /// <param name="source">The host elements to convert.</param>
    /// <param name="destination">The matching device slots.</param>
    private static void Pack(ReadOnlySpan<T> source, Span<int> destination)
    {
        if (typeof(T) == typeof(bool))
        {
            var values = MemoryMarshal.Cast<T, bool>(source);
            for (int i = 0; i < values.Length; i++)
                destination[i] = values[i] ? 1 : 0;
        }
        else
        {
            var values = MemoryMarshal.Cast<T, char>(source);
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
            if (mappedBuffer is not null)
                mappedBuffer.Download(offset, destination);
            else
                buffer!.CopyTo(offset, destination);
            return;
        }
        var padded = new int[destination.Length];
        paddedBuffer!.CopyTo(offset, padded);
        if (typeof(T) == typeof(bool))
        {
            var values = MemoryMarshal.Cast<T, bool>(destination.AsSpan());
            for (int i = 0; i < values.Length; i++)
                values[i] = padded[i] != 0;
        }
        else
        {
            var values = MemoryMarshal.Cast<T, char>(destination.AsSpan());
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
        buffer?.Dispose();
        paddedBuffer?.Dispose();
        mappedBuffer?.Dispose();
    }
}
