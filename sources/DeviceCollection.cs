using ILGPU;

namespace ParallelGpu;

/// <summary>Provides a device-compatible leaf or nested-array view.</summary>
/// <remarks>All leaves of the same element type share a packed buffer. Shape metadata stores
/// length, lower bound, and row-major stride per dimension. Invalid accesses defer an error for host validation.</remarks>
internal readonly struct DeviceCollection
{
    /// <summary>The leaf byte slice, or the complete packed buffer for a nested-array container.</summary>
    public readonly ArrayView<byte> Bytes;
    /// <summary>The flat element or child count; minus one denotes a null reference.</summary>
    public readonly int Length;
    /// <summary>The first dimension's length for views without shape metadata.</summary>
    public readonly int Rows;
    /// <summary>The second dimension's length for views without shape metadata.</summary>
    public readonly int Columns;
    /// <summary>The shared nested-array node table.</summary>
    public readonly ArrayView<int> RowMetadata;
    /// <summary>The first child descriptor's integer offset in the shared node table.</summary>
    public readonly int RowOffset;
    /// <summary>The shared dimension table storing length, lower bound, and stride triples.</summary>
    public readonly ArrayView<int> ShapeMetadata;
    /// <summary>The dimension-table offset, or minus one when Rows and Columns hold the shape.</summary>
    public readonly int ShapeOffset;
    /// <summary>The array rank; lists and flat views have rank one.</summary>
    public readonly int Rank;
    /// <summary>The invocation-local host reference identity; zero denotes null and minus one denotes a synthesized empty value view.</summary>
    public readonly int Identity;
    /// <summary>The invocation's deferred error and safe-reference context.</summary>
    public readonly DeviceExecutionContext Context;
    /// <summary>One for a read-only collection view; zero for a writable view.</summary>
    public readonly int IsReadOnly;
    /// <summary>The deferred exception code for an invalid flat index.</summary>
    public readonly int IndexErrorCode;
    /// <summary>The physical element stride in bytes; zero also marks a null-backed segment.</summary>
    public readonly int ElementSize;
    /// <summary>The invocation's shared write flags.</summary>
    public readonly ArrayView<int> Writes;
    /// <summary>The leaf's shared write-flag index, or minus one for a container.</summary>
    public readonly int DirtyIndex;
    /// <summary>The complete native list storage retained by an adapted list slice.</summary>
    public readonly ArrayView<byte> NativeListBytes;
    /// <summary>The complete native list length, used only when NativeListStorage is nonzero.</summary>
    public readonly int NativeListLength;
    /// <summary>One when this descriptor also represents its original native list storage.</summary>
    public readonly int NativeListStorage;
    /// <summary>The native list's write flag, which can differ from an adapted backing's flag.</summary>
    public readonly int NativeListDirtyIndex;

    /// <summary>Creates a leaf or nested-array view.</summary>
    /// <param name="bytes">The leaf slice or the complete packed device buffer.</param>
    /// <param name="length">The flat element or child count, or minus one for null.</param>
    /// <param name="rows">The first dimension's length for views without shape metadata.</param>
    /// <param name="columns">The second dimension's length for views without shape metadata.</param>
    /// <param name="rowMetadata">The shared nested-array metadata.</param>
    /// <param name="elementSize">The physical distance between leaf elements in bytes.</param>
    /// <param name="writes">The shared write flags.</param>
    /// <param name="dirtyIndex">The leaf's write-flag index, or minus one for a container.</param>
    /// <param name="shapeMetadata">The shared dimension metadata.</param>
    /// <param name="shapeOffset">The first dimension's metadata offset, or minus one for the inline shape.</param>
    /// <param name="rank">The array rank.</param>
    /// <param name="rowOffset">The first child's node metadata offset.</param>
    /// <param name="identity">The host reference identity; zero denotes null and minus one a synthesized empty value view.</param>
    /// <param name="context">The invocation's deferred error context.</param>
    /// <param name="readOnly">One for a read-only view, otherwise zero.</param>
    /// <param name="indexErrorCode">Three for array/span bounds, or nine for list/segment indexers.</param>
    /// <param name="nativeListBytes">The complete native list storage for an adapted list slice.</param>
    /// <param name="nativeListLength">The complete native list element count.</param>
    /// <param name="nativeListStorage">One when a native list projection is available, otherwise zero.</param>
    /// <param name="nativeListDirtyIndex">The write-flag index of the native list projection.</param>
    public DeviceCollection(ArrayView<byte> bytes, int length, int rows = -1, int columns = 0,
        ArrayView<int> rowMetadata = default, int elementSize = 0,
        ArrayView<int> writes = default, int dirtyIndex = -1,
        ArrayView<int> shapeMetadata = default, int shapeOffset = -1,
        int rank = 1, int rowOffset = 0, int identity = 0, DeviceExecutionContext context = default, int readOnly = 0,
        int indexErrorCode = 3, ArrayView<byte> nativeListBytes = default, int nativeListLength = 0,
        int nativeListStorage = 0, int nativeListDirtyIndex = -1)
    {
        Bytes = bytes;
        Length = length;
        Rows = rows < 0 ? length : rows;
        Columns = columns;
        RowMetadata = rowMetadata;
        RowOffset = rowOffset;
        ShapeMetadata = shapeMetadata;
        ShapeOffset = shapeOffset;
        Rank = rank;
        Identity = identity;
        Context = context;
        IsReadOnly = readOnly;
        IndexErrorCode = indexErrorCode;
        ElementSize = elementSize;
        Writes = writes;
        DirtyIndex = dirtyIndex;
        NativeListBytes = nativeListBytes;
        NativeListLength = nativeListLength;
        NativeListStorage = nativeListStorage;
        NativeListDirtyIndex = nativeListDirtyIndex;
    }

    /// <summary>Reads one flat leaf element without marking it as written.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The leaf view.</param>
    /// <param name="index">The zero-based flat index.</param>
    /// <returns>The element, or the default value after recording a null or bounds error.</returns>
    public static T Read<T>(DeviceCollection collection, int index) where T : unmanaged
    {
        if (!ValidIndex(collection, index))
            return default;
        return collection.Bytes.SubView((long)index * collection.ElementSize, collection.ElementSize).Cast<T>()[0];
    }

    /// <summary>Writes one flat element and marks the leaf for host copy-back.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The leaf view.</param>
    /// <param name="index">The zero-based flat index.</param>
    /// <param name="value">The value to store.</param>
    /// <remarks>Invalid or read-only accesses record a deferred error without changing the element.</remarks>
    public static void Write<T>(DeviceCollection collection, int index, T value) where T : unmanaged
    {
        if (!ValidIndex(collection, index))
            return;
        if (collection.IsReadOnly != 0)
        {
            collection.Context.Fail(5);
            return;
        }
        MarkWritten(collection);
        collection.Bytes.SubView((long)index * collection.ElementSize, collection.ElementSize).Cast<T>()[0] = value;
    }

    /// <summary>Returns a flat element reference and marks the leaf as potentially written.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The leaf view.</param>
    /// <param name="index">The zero-based flat index.</param>
    /// <returns>A reference to the device element, or work-item scratch storage after recording an invalid or read-only access.</returns>
    public static ref T Address<T>(DeviceCollection collection, int index) where T : unmanaged
    {
        if (!ValidIndex(collection, index))
            return ref collection.Context.ScratchAddress<T>();
        if (collection.IsReadOnly != 0)
        {
            collection.Context.Fail(5);
            return ref collection.Context.ScratchAddress<T>();
        }
        MarkWritten(collection);
        return ref collection.Bytes.SubView((long)index * collection.ElementSize, collection.ElementSize).Cast<T>()[0];
    }

    /// <summary>Returns a checked element reference without marking a read-only access as written.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The leaf view.</param>
    /// <param name="index">The zero-based flat index.</param>
    /// <returns>A reference for read-only access, or work-item scratch storage after recording a null or bounds error.</returns>
    public static ref T ReadOnlyAddress<T>(DeviceCollection collection, int index) where T : unmanaged
    {
        if (!ValidIndex(collection, index))
            return ref collection.Context.ScratchAddress<T>();
        return ref collection.Bytes.SubView((long)index * collection.ElementSize, collection.ElementSize).Cast<T>()[0];
    }

    /// <summary>Checks a flat access and records a deferred error for null or out-of-range references.</summary>
    /// <param name="collection">The collection reference.</param>
    /// <param name="index">The zero-based flat index.</param>
    /// <returns>Whether access is valid.</returns>
    private static bool ValidIndex(DeviceCollection collection, int index)
    {
        if (collection.Identity == 0)
        {
            collection.Context.Fail(2);
            return false;
        }
        if ((uint)index >= (uint)collection.Length)
        {
            collection.Context.Fail(collection.IndexErrorCode == 0 ? 3 : collection.IndexErrorCode);
            return false;
        }
        return true;
    }

    /// <summary>Checks a dimension query and records invalid or null accesses.</summary>
    /// <param name="collection">The array reference.</param>
    /// <param name="dimension">The zero-based dimension number.</param>
    /// <returns>Whether the query is valid.</returns>
    private static bool ValidDimension(DeviceCollection collection, int dimension)
    {
        if (collection.Identity == 0)
        {
            collection.Context.Fail(2);
            return false;
        }
        if ((uint)dimension >= (uint)collection.Rank)
        {
            collection.Context.Fail(3);
            return false;
        }
        return true;
    }

    /// <summary>Attaches the work item's context to a captured collection view.</summary>
    /// <param name="collection">The captured view.</param>
    /// <param name="context">The work item's deferred error and scratch context.</param>
    /// <returns>The view carrying that context.</returns>
    public static DeviceCollection WithContext(DeviceCollection collection, DeviceExecutionContext context) =>
        new(collection.Bytes, collection.Length, collection.Rows, collection.Columns,
            collection.RowMetadata, collection.ElementSize, collection.Writes, collection.DirtyIndex,
            collection.ShapeMetadata, collection.ShapeOffset, collection.Rank, collection.RowOffset,
            collection.Identity, context, collection.IsReadOnly, collection.IndexErrorCode,
            collection.NativeListBytes, collection.NativeListLength, collection.NativeListStorage,
            collection.NativeListDirtyIndex);

    /// <summary>Restores native list storage when an adapted list is used through a native declaration.</summary>
    /// <param name="collection">The native collection or adapted list slice.</param>
    /// <returns>The complete writable native list view, or the original descriptor for other collections.</returns>
    private static DeviceCollection NativeList(DeviceCollection collection) => collection.NativeListStorage == 0
        ? collection : new(collection.NativeListBytes, collection.NativeListLength,
            elementSize: collection.ElementSize, writes: collection.Writes, dirtyIndex: collection.NativeListDirtyIndex,
            identity: collection.Identity, context: collection.Context, indexErrorCode: 9);

    /// <summary>Returns the count selected by a native List, IList, or IReadOnlyList declaration.</summary>
    /// <param name="collection">The native collection or adapted list slice.</param>
    /// <returns>The complete native count, retaining normal null-reference validation.</returns>
    public static int NativeListCount(DeviceCollection collection) => Count(NativeList(collection));

    /// <summary>Reads a native list-interface element, including through an implicitly upcast adapted list.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The native collection or adapted list slice.</param>
    /// <param name="index">The native zero-based index.</param>
    /// <returns>The element, or the default value after recording a deferred access error.</returns>
    public static T NativeListRead<T>(DeviceCollection collection, int index) where T : unmanaged =>
        DeviceCollectionSlice.ReadIndexed<T>(NativeList(collection), index);

    /// <summary>Writes a native list-interface element, including through an implicitly upcast adapted list.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The native collection or adapted list slice.</param>
    /// <param name="index">The native zero-based index.</param>
    /// <param name="value">The value to store in the complete native list.</param>
    public static void NativeListWrite<T>(DeviceCollection collection, int index, T value) where T : unmanaged =>
        DeviceCollectionSlice.WriteIndexed(NativeList(collection), index, value);

    /// <summary>Atomically marks a leaf for copy-back.</summary>
    /// <param name="collection">The leaf view.</param>
    private static void MarkWritten(DeviceCollection collection)
    {
        if (collection.Writes[collection.DirtyIndex] == 0)
            Atomic.Exchange(ref collection.Writes[collection.DirtyIndex], 1);
    }

    /// <summary>Returns one dimension's contribution to the flat row-major offset.</summary>
    /// <param name="collection">The array view.</param>
    /// <param name="dimension">The zero-based dimension number.</param>
    /// <param name="index">The index in that dimension's native lower-bound convention.</param>
    /// <returns>The dimension's contribution in elements.</returns>
    public static int DimensionOffset(DeviceCollection collection, int dimension, int index)
    {
        if (!ValidDimension(collection, dimension))
            return 0;
        int length;
        int relative;
        int stride;
        if (collection.ShapeOffset < 0)
        {
            length = dimension == 0 ? collection.Rows : collection.Columns;
            relative = index;
            stride = dimension == 0 && collection.Columns != 0 ? collection.Columns : 1;
        }
        else
        {
            int position = collection.ShapeOffset + dimension * 3;
            length = collection.ShapeMetadata[position];
            relative = index - collection.ShapeMetadata[position + 1];
            stride = collection.ShapeMetadata[position + 2];
        }
        if ((uint)relative >= (uint)length)
        {
            collection.Context.Fail(3);
            return 0;
        }
        return relative * stride;
    }

    /// <summary>Reads a two-dimensional element.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The rectangular view.</param>
    /// <param name="row">The first index.</param>
    /// <param name="column">The second index.</param>
    /// <returns>The element.</returns>
    public static T Read2D<T>(DeviceCollection collection, int row, int column) where T : unmanaged =>
        Read<T>(collection, DimensionOffset(collection, 0, row) + DimensionOffset(collection, 1, column));

    /// <summary>Writes a two-dimensional element.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The rectangular view.</param>
    /// <param name="row">The first index.</param>
    /// <param name="column">The second index.</param>
    /// <param name="value">The value to store.</param>
    public static void Write2D<T>(DeviceCollection collection, int row, int column, T value) where T : unmanaged =>
        Write(collection, DimensionOffset(collection, 0, row) + DimensionOffset(collection, 1, column), value);

    /// <summary>Returns a two-dimensional element reference.</summary>
    /// <typeparam name="T">The device element type.</typeparam>
    /// <param name="collection">The rectangular view.</param>
    /// <param name="row">The first index.</param>
    /// <param name="column">The second index.</param>
    /// <returns>A reference to the element.</returns>
    public static ref T Address2D<T>(DeviceCollection collection, int row, int column) where T : unmanaged =>
        ref Address<T>(collection, DimensionOffset(collection, 0, row) + DimensionOffset(collection, 1, column));

    /// <summary>Resolves a nested child array, retaining its shape and shared identity.</summary>
    /// <param name="collection">The nested-array container.</param>
    /// <param name="index">The zero-based flat child index.</param>
    /// <returns>The child view, including a null view for a null child.</returns>
    public static DeviceCollection Row(DeviceCollection collection, int index)
    {
        if (!ValidIndex(collection, index))
            return WithContext(Null(), collection.Context);
        int position = collection.RowOffset + index * RowDescriptor.Width;
        int offset = collection.RowMetadata[position];
        int length = collection.RowMetadata[position + 1];
        int dirtyIndex = collection.RowMetadata[position + 2];
        int shapeOffset = collection.RowMetadata[position + 3];
        int rank = collection.RowMetadata[position + 4];
        int identity = collection.RowMetadata[position + 5];
        if (length < 0)
            return WithContext(Null(), collection.Context);
        bool leaf = dirtyIndex >= 0;
        return new(leaf ? collection.Bytes.SubView((long)offset * collection.ElementSize,
                (long)length * collection.ElementSize) : collection.Bytes,
            length, rowMetadata: collection.RowMetadata, elementSize: collection.ElementSize,
            writes: collection.Writes, dirtyIndex: dirtyIndex,
            shapeMetadata: collection.ShapeMetadata, shapeOffset: shapeOffset, rank: rank,
            rowOffset: leaf ? 0 : offset, identity: identity, context: collection.Context, readOnly: collection.IsReadOnly,
            indexErrorCode: collection.IndexErrorCode);
    }

    /// <summary>Creates the null collection reference representation.</summary>
    /// <returns>A null view.</returns>
    public static DeviceCollection Null() => new(default, -1);

    /// <summary>Checks a translated collection reference for null.</summary>
    /// <param name="collection">The collection reference.</param>
    /// <returns>Whether the reference is null.</returns>
    public static bool IsNull(DeviceCollection collection) => collection.Identity == 0;

    /// <summary>Compares translated collection reference identities.</summary>
    /// <param name="left">The first reference.</param>
    /// <param name="right">The second reference.</param>
    /// <returns>Whether both references denote the same host object or are null.</returns>
    public static bool Same(DeviceCollection left, DeviceCollection right) => left.Identity == right.Identity;

    /// <summary>Returns the flat element or child count.</summary>
    /// <param name="collection">The collection view.</param>
    /// <returns>The count.</returns>
    public static int Count(DeviceCollection collection)
    {
        if (collection.Identity != 0)
            return collection.Length;
        collection.Context.Fail(2);
        return 0;
    }

    /// <summary>Returns an array dimension's length.</summary>
    /// <param name="collection">The array view.</param>
    /// <param name="dimension">The zero-based dimension number.</param>
    /// <returns>The dimension length.</returns>
    public static int GetLength(DeviceCollection collection, int dimension)
    {
        if (!ValidDimension(collection, dimension))
            return 0;
        return collection.ShapeOffset < 0 ? dimension == 0 ? collection.Rows : collection.Columns
            : collection.ShapeMetadata[collection.ShapeOffset + dimension * 3];
    }

    /// <summary>Returns an array dimension's lower bound.</summary>
    /// <param name="collection">The array view.</param>
    /// <param name="dimension">The zero-based dimension number.</param>
    /// <returns>The lower bound.</returns>
    public static int GetLowerBound(DeviceCollection collection, int dimension)
    {
        if (!ValidDimension(collection, dimension))
            return 0;
        return collection.ShapeOffset < 0 ? 0 : collection.ShapeMetadata[collection.ShapeOffset + dimension * 3 + 1];
    }

    /// <summary>Returns an array dimension's inclusive upper bound.</summary>
    /// <param name="collection">The array view.</param>
    /// <param name="dimension">The zero-based dimension number.</param>
    /// <returns>The upper bound.</returns>
    public static int GetUpperBound(DeviceCollection collection, int dimension) =>
        GetLowerBound(collection, dimension) + GetLength(collection, dimension) - 1;

    /// <summary>Returns the array rank.</summary>
    /// <param name="collection">The array view.</param>
    /// <returns>The rank.</returns>
    public static int GetRank(DeviceCollection collection)
    {
        if (collection.Identity != 0)
            return collection.Rank;
        collection.Context.Fail(2);
        return 0;
    }

    /// <summary>Returns the flat count as a 64-bit value.</summary>
    /// <param name="collection">The collection view.</param>
    /// <returns>The count.</returns>
    public static long LongCount(DeviceCollection collection) => Count(collection);

    /// <summary>Returns a dimension length as a 64-bit value.</summary>
    /// <param name="collection">The array view.</param>
    /// <param name="dimension">The zero-based dimension number.</param>
    /// <returns>The dimension length.</returns>
    public static long GetLongLength(DeviceCollection collection, int dimension) => GetLength(collection, dimension);
}
