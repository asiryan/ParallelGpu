using ILGPU;

namespace ParallelGpu;

internal abstract class CapturedCollection
{
    internal abstract DeviceCollection View { get; }

    internal static bool IsNumeric(Type type) => type == typeof(float) || type == typeof(int) ||
        type == typeof(uint) || type == typeof(long) || type == typeof(ulong) ||
        type == typeof(double) || type == typeof(short) || type == typeof(ushort) ||
        type == typeof(byte) || type == typeof(sbyte);

    internal static Type? ElementType(Type type)
    {
        if (type.IsArray)
        {
            var element = type.GetElementType()!;
            if (type.GetArrayRank() <= 2 && IsNumeric(element))
                return element;
            if (type.GetArrayRank() == 1 && element.IsArray && element.GetArrayRank() == 1 &&
                IsNumeric(element.GetElementType()!))
                return element.GetElementType();
        }
        if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(List<>) ||
            type.GetGenericTypeDefinition() == typeof(IList<>) ||
            type.GetGenericTypeDefinition() == typeof(IReadOnlyList<>)) &&
            IsNumeric(type.GetGenericArguments()[0]))
            return type.GetGenericArguments()[0];
        return null;
    }
}

internal sealed class CapturedCollection<T> : CapturedCollection where T : unmanaged
{
    private readonly object source;
    private readonly CollectionTransfers transfers;
    private readonly CollectionBatch<T> batch;
    private readonly int rows = -1;
    private readonly int columns;

    internal T[] Snapshot { get; }
    internal int Offset { get; }
    internal int DirtyIndex { get; }
    internal RowDescriptor Row => new(Snapshot.Length == 0 ? 0 : Offset,
        Snapshot.Length, DirtyIndex);

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

    internal override DeviceCollection View => new(
        batch.View.SubView(Snapshot.Length == 0 ? 0 : Offset, Snapshot.Length).Cast<byte>(),
        Snapshot.Length, rows, columns, writes: transfers.Writes, dirtyIndex: DirtyIndex);

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
