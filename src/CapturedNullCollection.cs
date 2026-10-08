namespace ParallelGpu;

/// <summary>Preserves a null captured collection until its device reference is tested or dereferenced.</summary>
/// <param name="transfers">The invocation supplying its deferred error context.</param>
internal sealed class CapturedNullCollection(CollectionTransfers transfers) : CapturedCollection
{
    /// <inheritdoc/>
    internal override DeviceCollection View => DeviceCollection.WithContext(DeviceCollection.Null(), transfers.Context);

    /// <inheritdoc/>
    internal override RowDescriptor Row => RowDescriptor.Null;
}
