namespace ParallelGpu;

/// <summary>Exposes a host collection slice through the existing allocation and write flag of its backing array.</summary>
internal sealed class CapturedSliceCollection<T> : CapturedCollection
{
    private readonly CapturedCollection<T> backing;
    private readonly GpuCollectionSnapshot<T> snapshot;
    private readonly CollectionTransfers transfers;
    private readonly int shapeOffset;

    internal CapturedSliceCollection(CollectionTransfers transfers, CapturedCollection<T> backing, GpuCollectionSnapshot<T> snapshot)
    {
        this.transfers = transfers;
        this.backing = backing;
        this.snapshot = snapshot;
        shapeOffset = transfers.AddFlatShape(snapshot.Count);
    }

    internal override RowDescriptor Row => new(backing.Offset + snapshot.Offset, snapshot.Count,
        backing.DirtyIndex, shapeOffset, 1, Identity);

    internal override DeviceCollection View
    {
        get
        {
            var view = backing.View;
            return new(view.Bytes.SubView((long)snapshot.Offset * view.ElementSize, (long)snapshot.Count * view.ElementSize),
                snapshot.Count, elementSize: view.ElementSize, writes: view.Writes, dirtyIndex: view.DirtyIndex,
                shapeMetadata: transfers.Shapes, shapeOffset: shapeOffset, rank: 1, identity: Identity,
                context: view.Context, readOnly: snapshot.IsReadOnly ? 1 : 0);
        }
    }
}
