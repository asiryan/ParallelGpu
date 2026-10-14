namespace ParallelGpu;

/// <summary>Exposes a host collection slice through the existing allocation and write flag of its backing collection.</summary>
/// <remarks>Read-only status belongs to this alias; writable aliases still share the same backing storage.</remarks>
internal sealed class CapturedSliceCollection<T> : CapturedCollection
{
    private readonly CapturedCollection<T> backing;
    private readonly GpuCollectionSnapshot<T> snapshot;
    private readonly CollectionTransfers transfers;
    private readonly int shapeOffset;
    /// <summary>Distinguishes a default segment from a segment over an empty backing array.</summary>
    private readonly bool nullBackedSegment;
    /// <summary>Retains the CLR argument-range error contract of segment indexers.</summary>
    private readonly bool arraySegment;
    /// <summary>Retains the complete native representation of a list captured through its adapter.</summary>
    private readonly CapturedCollection<T>? nativeList;

    internal CapturedSliceCollection(CollectionTransfers transfers, CapturedCollection<T> backing, GpuCollectionSnapshot<T> snapshot,
        bool arraySegment = false, bool nullBackedSegment = false, CapturedCollection<T>? nativeList = null)
    {
        this.transfers = transfers;
        this.backing = backing;
        this.snapshot = snapshot;
        this.arraySegment = arraySegment;
        this.nullBackedSegment = nullBackedSegment;
        this.nativeList = nativeList;
        shapeOffset = transfers.AddFlatShape(snapshot.Count);
    }

    internal override RowDescriptor Row => new(backing.Offset + snapshot.Offset, snapshot.Count,
        backing.DirtyIndex, shapeOffset, 1, Identity);

    internal override DeviceCollection View
    {
        get
        {
            var view = backing.View;
            var nativeView = nativeList?.View ?? default;
            // ILGPU rejects SubView(end, 0); empty slices never dereference byte storage.
            return new(snapshot.Count == 0 ? default :
                view.Bytes.SubView((long)snapshot.Offset * view.ElementSize, (long)snapshot.Count * view.ElementSize),
                snapshot.Count, elementSize: nullBackedSegment ? 0 : view.ElementSize, writes: view.Writes, dirtyIndex: view.DirtyIndex,
                shapeMetadata: transfers.Shapes, shapeOffset: shapeOffset, rank: 1, identity: Identity,
                context: view.Context, readOnly: snapshot.IsReadOnly ? 1 : 0,
                indexErrorCode: arraySegment ? 9 : view.IndexErrorCode,
                nativeListBytes: nativeView.Bytes, nativeListLength: nativeView.Length,
                nativeListStorage: nativeList is null ? 0 : 1, nativeListDirtyIndex: nativeView.DirtyIndex);
        }
    }
}
