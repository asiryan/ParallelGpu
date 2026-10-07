using ILGPU;

namespace ParallelGpu;

internal readonly struct RowDescriptor(int offset, int length, int dirtyIndex)
{
    public readonly int Offset = offset;
    public readonly int Length = length;
    public readonly int DirtyIndex = dirtyIndex;
}

// Jagged rows refer to slices of the same packed data buffer as captured flat arrays.
internal readonly struct DeviceCollection
{
    public readonly ArrayView<byte> Bytes;
    public readonly int Length;
    public readonly int Rows;
    public readonly int Columns;
    public readonly ArrayView<int> RowMetadata;
    public readonly int ElementSize;
    public readonly ArrayView<int> Writes;
    public readonly int DirtyIndex;

    public DeviceCollection(ArrayView<byte> bytes, int length, int rows = -1, int columns = 0,
        ArrayView<int> rowMetadata = default, int elementSize = 0,
        ArrayView<int> writes = default, int dirtyIndex = -1)
    {
        Bytes = bytes;
        Length = length;
        Rows = rows < 0 ? length : rows;
        Columns = columns;
        RowMetadata = rowMetadata;
        ElementSize = elementSize;
        Writes = writes;
        DirtyIndex = dirtyIndex;
    }

    public static T Read<T>(DeviceCollection collection, int index) where T : unmanaged =>
        collection.Bytes.Cast<T>()[index];

    public static void Write<T>(DeviceCollection collection, int index, T value) where T : unmanaged
    {
        MarkWritten(collection);
        collection.Bytes.Cast<T>()[index] = value;
    }

    public static ref T Address<T>(DeviceCollection collection, int index) where T : unmanaged
    {
        // A returned element reference may be modified by a scalar helper.
        MarkWritten(collection);
        return ref collection.Bytes.Cast<T>()[index];
    }

    private static void MarkWritten(DeviceCollection collection)
    {
        // Flags only change from zero to one; a stale zero merely repeats the atomic write.
        if (collection.Writes[collection.DirtyIndex] == 0)
            Atomic.Exchange(ref collection.Writes[collection.DirtyIndex], 1);
    }

    public static T Read2D<T>(DeviceCollection collection, int row, int column) where T : unmanaged =>
        Read<T>(collection, row * collection.Columns + column);

    public static void Write2D<T>(DeviceCollection collection, int row, int column, T value) where T : unmanaged =>
        Write(collection, row * collection.Columns + column, value);

    public static ref T Address2D<T>(DeviceCollection collection, int row, int column) where T : unmanaged =>
        ref Address<T>(collection, row * collection.Columns + column);

    public static DeviceCollection Row(DeviceCollection collection, int index)
    {
        int position = index * 3;
        int offset = collection.RowMetadata[position];
        int length = collection.RowMetadata[position + 1];
        int dirtyIndex = collection.RowMetadata[position + 2];
        return new(collection.Bytes.SubView((long)offset * collection.ElementSize,
            (long)length * collection.ElementSize), length, writes: collection.Writes, dirtyIndex: dirtyIndex);
    }

    public static int Count(DeviceCollection collection) => collection.Length;

    public static int GetLength(DeviceCollection collection, int dimension) =>
        dimension == 0 ? collection.Rows : collection.Columns;

    public static long LongCount(DeviceCollection collection) => collection.Length;

    public static long GetLongLength(DeviceCollection collection, int dimension) =>
        GetLength(collection, dimension);
}
