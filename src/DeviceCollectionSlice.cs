namespace ParallelGpu;

/// <summary>Provides device operations for contiguous memory, spans, and array segments.</summary>
internal static class DeviceCollectionSlice
{
    /// <summary>Initializes an empty value view while retaining the work item error context.</summary>
    public static void Initialize(ref DeviceCollection collection, DeviceExecutionContext context) =>
        collection = new(default, 0, identity: -1, context: context);

    /// <summary>Normalizes a default or null-backed memory value to an empty view with span indexing.</summary>
    public static DeviceCollection Normalize(DeviceCollection collection) => collection.Identity == 0
        ? new(default, 0, elementSize: collection.ElementSize, identity: -1, context: collection.Context,
            readOnly: collection.IsReadOnly)
        : collection.IndexErrorCode == 3 ? collection : WithIndexError(collection, 3);

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

    /// <summary>Creates a whole array segment; a null array becomes the default segment.</summary>
    public static DeviceCollection Segment(DeviceCollection collection) =>
        Indexed(Normalize(collection));

    /// <summary>Constructs a whole array segment; its array must be nonnull.</summary>
    public static DeviceCollection SegmentConstructorWhole(DeviceCollection collection)
    {
        if (collection.Identity == 0)
            collection.Context.Fail(7);
        return Segment(collection);
    }

    /// <summary>Creates a bounded array segment; its array must be nonnull.</summary>
    public static DeviceCollection SegmentConstructor(DeviceCollection collection, int start, int length)
    {
        if (collection.Identity == 0)
        {
            collection.Context.Fail(7);
            return Segment(collection);
        }
        return Indexed(Slice(collection, start, length));
    }

    /// <summary>Creates a segment slice, rejecting a default segment with no backing array.</summary>
    public static DeviceCollection SegmentSlice(DeviceCollection collection, int start, int length)
    {
        if (collection.ElementSize == 0)
        {
            collection.Context.Fail(8);
            return Segment(collection);
        }
        return Indexed(Slice(collection, start, length));
    }

    /// <summary>Creates a segment slice continuing to the end of its backing region.</summary>
    public static DeviceCollection SegmentSliceFrom(DeviceCollection collection, int start) =>
        SegmentSlice(collection, start, Count(collection) - start);

    /// <summary>Uses the argument-range exception required by collection integer indexers.</summary>
    public static DeviceCollection Indexed(DeviceCollection collection) =>
        collection.IndexErrorCode == 9 ? collection : WithIndexError(collection, 9);

    /// <summary>Reads a List or list-interface indexer without changing its null semantics.</summary>
    public static T ReadIndexed<T>(DeviceCollection collection, int index) where T : unmanaged
    {
        if (collection.Identity != 0 && collection.ElementSize == 0)
        {
            collection.Context.Fail(8);
            return default;
        }
        return DeviceCollection.Read<T>(Indexed(collection), index);
    }

    /// <summary>Writes a List or list-interface indexer without changing its null semantics.</summary>
    public static void WriteIndexed<T>(DeviceCollection collection, int index, T value) where T : unmanaged
    {
        if (collection.Identity != 0 && collection.ElementSize == 0)
        {
            collection.Context.Fail(8);
            return;
        }
        DeviceCollection.Write(Indexed(collection), index, value);
    }

    /// <summary>Copies a descriptor while replacing its flat-index error behavior.</summary>
    private static DeviceCollection WithIndexError(DeviceCollection collection, int code) =>
        new(collection.Bytes, collection.Length, collection.Rows, collection.Columns,
            collection.RowMetadata, collection.ElementSize, collection.Writes, collection.DirtyIndex,
            collection.ShapeMetadata, collection.ShapeOffset, collection.Rank, collection.RowOffset,
            collection.Identity, collection.Context, collection.IsReadOnly, code);
}
