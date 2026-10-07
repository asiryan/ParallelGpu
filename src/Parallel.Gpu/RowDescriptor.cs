namespace ParallelGpu;

/// <summary>Describes a jagged row's slice in a packed numeric buffer.</summary>
/// <param name="offset">The row's element offset in the numeric buffer.</param>
/// <param name="length">The number of row elements.</param>
/// <param name="dirtyIndex">The index of the row's shared write flag.</param>
internal readonly struct RowDescriptor(int offset, int length, int dirtyIndex)
{
    /// <summary>The row's element offset in the packed numeric buffer.</summary>
    public readonly int Offset = offset;
    /// <summary>The number of elements in the row.</summary>
    public readonly int Length = length;
    /// <summary>The write-flag index shared by all aliases of the row.</summary>
    public readonly int DirtyIndex = dirtyIndex;
}
