namespace ParallelGpu;

/// <summary>Represents a jagged array through shared leaf slices and packed row metadata.</summary>
/// <typeparam name="T">The supported row element type.</typeparam>
/// <remarks>Repeated row references use the same leaf slice and write flag.</remarks>
internal sealed class CapturedJaggedCollection<T> : CapturedCollection where T : unmanaged
{
    /// <summary>The invocation registry containing shared row metadata and write flags.</summary>
    private readonly CollectionTransfers transfers;
    /// <summary>The typed batch containing all row slices, including rows also captured independently.</summary>
    private readonly CollectionBatch<T> batch;
    /// <summary>The integer offset of this jagged array's first descriptor in the shared row table.</summary>
    private readonly int rowOffset;
    /// <summary>The number of rows represented by this jagged container.</summary>
    private readonly int rowCount;

    /// <summary>Registers each row and reserves the jagged array's metadata range.</summary>
    /// <param name="transfers">The transfer registry for the current invocation.</param>
    /// <param name="source">The jagged array; ragged and empty rows are supported.</param>
    /// <exception cref="ArgumentException">The array contains a null row.</exception>
    internal CapturedJaggedCollection(CollectionTransfers transfers, T[][] source)
    {
        this.transfers = transfers;
        if (source.Any(row => row is null))
            throw new ArgumentException("Captured jagged arrays cannot contain null rows.", "body");
        // Register leaf rows before building descriptors so aliases reuse both offsets and write flags.
        var descriptors = new RowDescriptor[source.Length];
        for (int i = 0; i < source.Length; i++)
            descriptors[i] = ((CapturedCollection<T>)transfers.Add<T>(source[i])).Row;
        rowOffset = transfers.AddRows(descriptors);
        rowCount = source.Length;
        batch = transfers.Batch<T>();
    }

    /// <inheritdoc/>
    internal override DeviceCollection View => new(batch.View, rowCount,
        rowMetadata: rowCount == 0 ? default : transfers.Rows.SubView(rowOffset, checked(rowCount * 3)),
        elementSize: batch.ElementSize, writes: transfers.Writes);
}
