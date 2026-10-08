using ILGPU;

namespace ParallelGpu;

/// <summary>Provides a device-compatible representation of a captured collection.</summary>
/// <remarks>Jagged rows refer to slices of the same packed buffer as captured flat arrays.
/// Kernel access assumes valid indices. Write flags also track references that may be modified.</remarks>
internal readonly struct DeviceCollection
{
    /// <summary>The byte view of the collection slice, or the complete packed buffer for jagged arrays.</summary>
    public readonly ArrayView<byte> Bytes;
    /// <summary>The flat element count, or the number of rows for a jagged array.</summary>
    public readonly int Length;
    /// <summary>The first dimension's length.</summary>
    public readonly int Rows;
    /// <summary>The second dimension's length for a rectangular array.</summary>
    public readonly int Columns;
    /// <summary>The jagged row table, with an element offset, length, and write-flag index per row.</summary>
    public readonly ArrayView<int> RowMetadata;
    /// <summary>The physical element stride in bytes, including padding for Boolean and character elements.</summary>
    public readonly int ElementSize;
    /// <summary>The invocation's device write flags.</summary>
    public readonly ArrayView<int> Writes;
    /// <summary>The write-flag index for this leaf collection, or minus one for a jagged container.</summary>
    public readonly int DirtyIndex;

    /// <summary>Creates a view of a leaf collection or a jagged container.</summary>
    /// <param name="bytes">The leaf slice or the complete packed device buffer.</param>
    /// <param name="length">The leaf element count or jagged row count.</param>
    /// <param name="rows">The rectangular row count, or minus one to use the collection length.</param>
    /// <param name="columns">The rectangular column count.</param>
    /// <param name="rowMetadata">The jagged row metadata slice.</param>
    /// <param name="elementSize">The physical element stride in bytes for flat access and jagged row lookup.</param>
    /// <param name="writes">The shared device write flags.</param>
    /// <param name="dirtyIndex">The leaf's write-flag index, or minus one for a jagged container.</param>
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

    /// <summary>Reads one flat element without marking the collection as written.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="collection">The device collection view.</param>
    /// <param name="index">The zero-based element index.</param>
    /// <returns>The element value.</returns>
    public static T Read<T>(DeviceCollection collection, int index) where T : unmanaged =>
        collection.Bytes.SubView((long)index * collection.ElementSize, collection.ElementSize).Cast<T>()[0];

    /// <summary>Writes one flat element and marks the collection for host copy-back.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="collection">The device collection view.</param>
    /// <param name="index">The zero-based element index.</param>
    /// <param name="value">The value to store.</param>
    public static void Write<T>(DeviceCollection collection, int index, T value) where T : unmanaged
    {
        MarkWritten(collection);
        collection.Bytes.SubView((long)index * collection.ElementSize, collection.ElementSize).Cast<T>()[0] = value;
    }

    /// <summary>Returns an element reference and marks its collection as potentially written.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="collection">The device collection view.</param>
    /// <param name="index">The zero-based element index.</param>
    /// <returns>A reference to the device element.</returns>
    /// <remarks>Copy-back is conservative because a scalar helper may modify the returned reference.</remarks>
    public static ref T Address<T>(DeviceCollection collection, int index) where T : unmanaged
    {
        // A returned element reference may be modified by a scalar helper.
        MarkWritten(collection);
        return ref collection.Bytes.SubView((long)index * collection.ElementSize, collection.ElementSize).Cast<T>()[0];
    }

    /// <summary>Atomically marks a leaf collection for host copy-back.</summary>
    /// <param name="collection">The leaf collection whose flag is updated.</param>
    private static void MarkWritten(DeviceCollection collection)
    {
        // Flags only change from zero to one; a stale zero merely repeats the atomic write.
        if (collection.Writes[collection.DirtyIndex] == 0)
            Atomic.Exchange(ref collection.Writes[collection.DirtyIndex], 1);
    }

    /// <summary>Reads a rectangular array element using row-major addressing.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="collection">The device collection view.</param>
    /// <param name="row">The zero-based row index.</param>
    /// <param name="column">The zero-based column index.</param>
    /// <returns>The element value.</returns>
    public static T Read2D<T>(DeviceCollection collection, int row, int column) where T : unmanaged =>
        Read<T>(collection, row * collection.Columns + column);

    /// <summary>Writes a rectangular array element and marks the array for host copy-back.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="collection">The device collection view.</param>
    /// <param name="row">The zero-based row index.</param>
    /// <param name="column">The zero-based column index.</param>
    /// <param name="value">The value to store.</param>
    public static void Write2D<T>(DeviceCollection collection, int row, int column, T value) where T : unmanaged =>
        Write(collection, row * collection.Columns + column, value);

    /// <summary>Returns a rectangular element reference and marks the array as potentially written.</summary>
    /// <typeparam name="T">The supported element type.</typeparam>
    /// <param name="collection">The device collection view.</param>
    /// <param name="row">The zero-based row index.</param>
    /// <param name="column">The zero-based column index.</param>
    /// <returns>A reference to the device element.</returns>
    public static ref T Address2D<T>(DeviceCollection collection, int row, int column) where T : unmanaged =>
        ref Address<T>(collection, row * collection.Columns + column);

    /// <summary>Resolves a jagged row into its leaf view and shared write flag.</summary>
    /// <param name="collection">The jagged container view.</param>
    /// <param name="index">The zero-based row index.</param>
    /// <returns>The row's device view.</returns>
    public static DeviceCollection Row(DeviceCollection collection, int index)
    {
        int position = index * 3;
        int offset = collection.RowMetadata[position];
        int length = collection.RowMetadata[position + 1];
        int dirtyIndex = collection.RowMetadata[position + 2];
        return new(collection.Bytes.SubView((long)offset * collection.ElementSize,
            (long)length * collection.ElementSize), length, elementSize: collection.ElementSize,
            writes: collection.Writes, dirtyIndex: dirtyIndex);
    }

    /// <summary>Returns a collection's length for translated array and list access.</summary>
    /// <param name="collection">The device collection view.</param>
    /// <returns>The element count, or jagged row count.</returns>
    public static int Count(DeviceCollection collection) => collection.Length;

    /// <summary>Returns a rectangular array dimension's length.</summary>
    /// <param name="collection">The device collection view.</param>
    /// <param name="dimension">Zero for rows or one for columns.</param>
    /// <returns>The selected dimension's length.</returns>
    public static int GetLength(DeviceCollection collection, int dimension) =>
        dimension == 0 ? collection.Rows : collection.Columns;

    /// <summary>Returns the collection length as a 64-bit value.</summary>
    /// <param name="collection">The device collection view.</param>
    /// <returns>The element count, or jagged row count.</returns>
    public static long LongCount(DeviceCollection collection) => collection.Length;

    /// <summary>Returns a rectangular array dimension's length as a 64-bit value.</summary>
    /// <param name="collection">The device collection view.</param>
    /// <param name="dimension">Zero for rows or one for columns.</param>
    /// <returns>The selected dimension's length.</returns>
    public static long GetLongLength(DeviceCollection collection, int dimension) =>
        GetLength(collection, dimension);
}
