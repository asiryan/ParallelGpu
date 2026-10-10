namespace ParallelGpu;

/// <summary>Represents nested arrays of arbitrary depth through shared leaf and container descriptors.</summary>
/// <typeparam name="T">The supported ultimate leaf element type.</typeparam>
/// <remarks>Aliases at every depth share one descriptor identity. Null children and empty
/// containers retain their distinction. Array topology is fixed during device execution.</remarks>
internal sealed class CapturedJaggedCollection<T> : CapturedCollection
{
    /// <summary>The invocation's metadata and write flags.</summary>
    private readonly CollectionTransfers transfers;
    /// <summary>The batch containing all ultimate leaves.</summary>
    private readonly CollectionBatch<T> batch;
    /// <summary>The first child's metadata offset.</summary>
    private readonly int rowOffset;
    /// <summary>The flat child count.</summary>
    private readonly int rowCount;
    /// <summary>The container's dimension metadata offset.</summary>
    private readonly int shapeOffset;
    /// <summary>The container's array rank.</summary>
    private readonly int rank;

    /// <summary>Registers every nonnull child and reserves this container's metadata.</summary>
    /// <param name="transfers">The invocation's transfer registry.</param>
    /// <param name="source">The array whose elements are arrays.</param>
    internal CapturedJaggedCollection(CollectionTransfers transfers, Array source)
    {
        this.transfers = transfers;
        rank = source.Rank;
        shapeOffset = transfers.AddShape(source);
        var descriptors = new RowDescriptor[source.Length];
        int index = 0;
        foreach (var child in source)
            descriptors[index++] = child is null ? RowDescriptor.Null : transfers.Add<T>(child).Row;
        rowOffset = transfers.AddRows(descriptors);
        rowCount = source.Length;
        batch = transfers.Batch<T>();
    }

    /// <inheritdoc/>
    internal override RowDescriptor Row => new(rowOffset, rowCount, -1, shapeOffset, rank, Identity);

    /// <inheritdoc/>
    internal override DeviceCollection View => new(batch.View, rowCount,
        rowMetadata: transfers.Rows, elementSize: batch.ElementSize, writes: transfers.Writes,
        shapeMetadata: transfers.Shapes, shapeOffset: shapeOffset, rank: rank,
        rowOffset: rowOffset, identity: Identity, context: transfers.Context);
}
