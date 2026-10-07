using ILGPU;

namespace ParallelGpu;

/// <summary>Maps one flat array, list, or rectangular array to a slice of a packed numeric buffer.</summary>
/// <typeparam name="T">The numeric element type.</typeparam>
internal sealed class CapturedCollection<T> : CapturedCollection where T : unmanaged
{
    /// <summary>The original host collection updated when this slice is copied back.</summary>
    private readonly object source;
    /// <summary>The invocation registry providing the shared write flags.</summary>
    private readonly CollectionTransfers transfers;
    /// <summary>The typed batch containing this collection's packed device slice.</summary>
    private readonly CollectionBatch<T> batch;
    /// <summary>The rectangular row count, or minus one for a flat collection.</summary>
    private readonly int rows = -1;
    /// <summary>The rectangular column count used for row-major addressing.</summary>
    private readonly int columns;

    /// <summary>Gets the flat host data used when packing the device buffer.</summary>
    /// <remarks>Flat arrays are used directly; lists and rectangular arrays are flattened into snapshots.</remarks>
    internal T[] Snapshot { get; }
    /// <summary>Gets the element offset of this collection in its packed numeric buffer.</summary>
    internal int Offset { get; }
    /// <summary>Gets the index of this collection's device write flag.</summary>
    internal int DirtyIndex { get; }
    /// <summary>Gets row metadata that preserves aliases to this collection in jagged arrays.</summary>
    internal RowDescriptor Row => new(Snapshot.Length == 0 ? 0 : Offset,
        Snapshot.Length, DirtyIndex);

    /// <summary>Snapshots a host collection and reserves its buffer slice and write flag.</summary>
    /// <param name="transfers">The transfer registry for the current invocation.</param>
    /// <param name="source">A numeric array, list, or zero-based rectangular array.</param>
    /// <exception cref="NotSupportedException">The collection type or array lower bounds are unsupported.</exception>
    internal CapturedCollection(CollectionTransfers transfers, object source)
    {
        this.transfers = transfers;
        this.source = source;
        switch (source)
        {
            case T[] array:
                Snapshot = array;
                break;
            case List<T> list:
                Snapshot = list.ToArray();
                break;
            case T[,] matrix:
                if (matrix.GetLowerBound(0) != 0 || matrix.GetLowerBound(1) != 0)
                    throw new NotSupportedException("Captured rectangular arrays must have zero lower bounds.");
                rows = matrix.GetLength(0);
                columns = matrix.GetLength(1);
                Snapshot = new T[matrix.Length];
                for (int row = 0; row < rows; row++)
                    for (int column = 0; column < columns; column++)
                        Snapshot[row * columns + column] = matrix[row, column];
                break;
            default:
                throw new NotSupportedException(
                    "Captured collections must be arrays or List<T>, including when typed as IList<T> or IReadOnlyList<T>.");
        }
        batch = transfers.Batch<T>();
        DirtyIndex = transfers.NextCollection();
        Offset = batch.Add(this);
    }

    /// <inheritdoc/>
    internal override DeviceCollection View => new(
        batch.View.SubView(Snapshot.Length == 0 ? 0 : Offset, Snapshot.Length).Cast<byte>(),
        Snapshot.Length, rows, columns, writes: transfers.Writes, dirtyIndex: DirtyIndex);

    /// <summary>Writes downloaded elements into the original host collection.</summary>
    /// <param name="values">The downloaded slice containing exactly this collection's elements.</param>
    /// <exception cref="InvalidOperationException">A captured list changed length during execution.</exception>
    internal void Apply(ReadOnlySpan<T> values)
    {
        switch (source)
        {
            case T[] array:
                values.CopyTo(array);
                break;
            case List<T> list:
                if (list.Count != Snapshot.Length)
                    throw new InvalidOperationException("A captured list changed length while the loop was running.");
                for (int i = 0; i < values.Length; i++)
                    list[i] = values[i];
                break;
            case T[,] matrix:
                for (int row = 0; row < rows; row++)
                    for (int column = 0; column < columns; column++)
                        matrix[row, column] = values[row * columns + column];
                break;
        }
    }
}
