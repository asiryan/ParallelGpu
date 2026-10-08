namespace ParallelGpu;

/// <summary>Describes one nested-array child in a shared metadata table.</summary>
/// <param name="offset">The leaf's element offset or the container's child-table offset.</param>
/// <param name="length">The flat element or child count, or minus one for null.</param>
/// <param name="dirtyIndex">The shared leaf write-flag index, or minus one for a container.</param>
/// <param name="shapeOffset">The shape metadata offset, or -1 for an inline zero-based flat shape.</param>
/// <param name="rank">The array rank.</param>
/// <param name="identity">The invocation-local host reference identity.</param>
internal readonly struct RowDescriptor(int offset, int length, int dirtyIndex, int shapeOffset, int rank, int identity)
{
    /// <summary>The number of integers in each packed descriptor.</summary>
    internal const int Width = 6;
    /// <summary>The leaf data or child-table offset.</summary>
    public readonly int Offset = offset;
    /// <summary>The flat count; minus one denotes null.</summary>
    public readonly int Length = length;
    /// <summary>The leaf's shared write-flag index, or minus one for a container.</summary>
    public readonly int DirtyIndex = dirtyIndex;
    /// <summary>The shape metadata offset, or -1 for an inline zero-based flat shape.</summary>
    public readonly int ShapeOffset = shapeOffset;
    /// <summary>The array rank.</summary>
    public readonly int Rank = rank;
    /// <summary>The shared host reference identity.</summary>
    public readonly int Identity = identity;
    /// <summary>The null child descriptor.</summary>
    internal static RowDescriptor Null => new(0, -1, -1, -1, 1, 0);
}
