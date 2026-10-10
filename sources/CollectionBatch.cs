namespace ParallelGpu;

/// <summary>Defines allocation and copy-back operations for a batch of one supported element type.</summary>
internal abstract class CollectionBatch : IDisposable
{
    /// <summary>Uploads registered host collections into a shared device allocation.</summary>
    /// <param name="gpu">The executor that owns the resulting allocation.</param>
    internal abstract void Upload(Gpu gpu);
    /// <summary>Downloads written collection slices and updates their original host objects.</summary>
    /// <param name="written">The write flags read after device execution.</param>
    internal abstract void CopyBack(int[] written);
    /// <summary>Releases the batch's device allocation.</summary>
    public abstract void Dispose();
}
