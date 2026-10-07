using ILGPU;

namespace ParallelGpu;

internal sealed class CapturedJaggedCollection<T> : CapturedCollection where T : unmanaged
{
    private readonly CollectionTransfers transfers;
    private readonly CollectionBatch<T> batch;
    private readonly int rowOffset;
    private readonly int rowCount;

    internal CapturedJaggedCollection(CollectionTransfers transfers, T[][] source)
    {
        this.transfers = transfers;
        if (source.Any(row => row is null))
            throw new ArgumentException("Captured jagged arrays cannot contain null rows.", "body");
        var descriptors = new RowDescriptor[source.Length];
        for (int i = 0; i < source.Length; i++)
            descriptors[i] = ((CapturedCollection<T>)transfers.Add<T>(source[i])).Row;
        rowOffset = transfers.AddRows(descriptors);
        rowCount = source.Length;
        batch = transfers.Batch<T>();
    }

    internal override DeviceCollection View => new(batch.View.Cast<byte>(), rowCount,
        rowMetadata: rowCount == 0 ? default : transfers.Rows.SubView(rowOffset, checked(rowCount * 3)),
        elementSize: Interop.SizeOf<T>(), writes: transfers.Writes);
}
