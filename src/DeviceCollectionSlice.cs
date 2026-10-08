namespace ParallelGpu;

/// <summary>Provides device operations for contiguous memory and span descriptors.</summary>
internal static class DeviceCollectionSlice
{
    /// <summary>Initializes an empty value view while retaining the work item error context.</summary>
    public static void Initialize(ref DeviceCollection collection, DeviceExecutionContext context) =>
        collection = new(default, 0, identity: -1, context: context);

    /// <summary>Normalizes a default or null-backed memory value to an empty nonnull view.</summary>
    public static DeviceCollection Normalize(DeviceCollection collection) => collection.Identity == 0
        ? new(default, 0, elementSize: collection.ElementSize, identity: -1, context: collection.Context,
            readOnly: collection.IsReadOnly)
        : collection;

    /// <summary>Returns the length of a memory, span, or segment value, including its default value.</summary>
    public static int Count(DeviceCollection collection) => Normalize(collection).Length;

    /// <summary>Returns whether a memory or span view is empty.</summary>
    public static bool IsEmpty(DeviceCollection collection) => Count(collection) == 0;

    /// <summary>Returns a view that forbids device writes through this alias.</summary>
    public static DeviceCollection ReadOnly(DeviceCollection collection)
    {
        collection = Normalize(collection);
        return new(collection.Bytes, collection.Length, collection.Rows, collection.Columns,
            collection.RowMetadata, collection.ElementSize, collection.Writes, collection.DirtyIndex,
            collection.ShapeMetadata, collection.ShapeOffset, collection.Rank, collection.RowOffset,
            collection.Identity, collection.Context, 1);
    }

    /// <summary>Creates a validated zero-based slice of contiguous memory.</summary>
    public static DeviceCollection Slice(DeviceCollection collection, int start, int length)
    {
        collection = Normalize(collection);
        if ((uint)start > (uint)collection.Length || (uint)length > (uint)(collection.Length - start))
        {
            collection.Context.Fail(6);
            return new(default, 0, elementSize: collection.ElementSize, identity: -1,
                context: collection.Context, readOnly: collection.IsReadOnly);
        }
        return new(collection.Bytes.SubView((long)start * collection.ElementSize, (long)length * collection.ElementSize),
            length, elementSize: collection.ElementSize, writes: collection.Writes, dirtyIndex: collection.DirtyIndex,
            identity: collection.Identity, context: collection.Context, readOnly: collection.IsReadOnly);
    }

    /// <summary>Creates a slice continuing to the end of contiguous memory.</summary>
    public static DeviceCollection SliceFrom(DeviceCollection collection, int start) =>
        Slice(collection, start, Count(collection) - start);
}
