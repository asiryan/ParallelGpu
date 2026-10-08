using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;

namespace ParallelGpu;

/// <summary>Maps a leaf array of any rank or a list to a packed device-buffer slice.</summary>
/// <typeparam name="T">The supported host leaf element type.</typeparam>
internal sealed class CapturedCollection<T> : CapturedCollection
{
    /// <summary>Returns a reference to an array's physical first element.</summary>
    /// <param name="array">The nonempty leaf array.</param>
    /// <returns>A reference to its first element.</returns>
    private delegate ref T FirstElement(Array array);
    /// <summary>Rank-specific host address getters, shared across invocations.</summary>
    private static readonly ConcurrentDictionary<Type, FirstElement> firstElements = new();
    /// <summary>The original host collection.</summary>
    private readonly object source;
    /// <summary>The invocation's shared metadata and write flags.</summary>
    private readonly CollectionTransfers transfers;
    /// <summary>The batch containing this leaf's packed slice.</summary>
    private readonly CollectionBatch<T> batch;
    /// <summary>The dimension metadata offset.</summary>
    private readonly int shapeOffset;
    /// <summary>The leaf array rank; lists have rank one.</summary>
    private readonly int rank;

    /// <summary>Gets the flattened data; native flat arrays are used directly.</summary>
    internal T[] Snapshot { get; }
    /// <summary>Gets the element offset in the packed buffer.</summary>
    internal int Offset { get; }
    /// <summary>Gets the shared write-flag index.</summary>
    internal int DirtyIndex { get; }
    /// <inheritdoc/>
    internal override RowDescriptor Row => new(Snapshot.Length == 0 ? 0 : Offset,
        Snapshot.Length, DirtyIndex, shapeOffset, rank, Identity);

    /// <summary>Snapshots a host leaf and reserves its shape, data slice, and write flag.</summary>
    /// <param name="transfers">The invocation's transfer registry.</param>
    /// <param name="source">An array of any rank with supported leaf elements, or a list.</param>
    /// <exception cref="NotSupportedException">The runtime collection is unsupported.</exception>
    internal CapturedCollection(CollectionTransfers transfers, object source)
    {
        this.transfers = transfers;
        this.source = source;
        switch (source)
        {
            case T[] array:
                Snapshot = array;
                rank = 1;
                shapeOffset = transfers.AddShape(array);
                break;
            case List<T> list:
                Snapshot = list.ToArray();
                rank = 1;
                shapeOffset = transfers.AddFlatShape(list.Count);
                break;
            case Array array when array.GetType().GetElementType() == typeof(T):
                rank = array.Rank;
                shapeOffset = transfers.AddShape(array);
                Snapshot = array.Length == 0 ? Array.Empty<T>() :
                    MemoryMarshal.CreateReadOnlySpan(ref First(array), array.Length).ToArray();
                break;
            default:
                throw new NotSupportedException(
                    "Captured collections must be arrays or List<T>, including when typed as IList<T> or IReadOnlyList<T>.");
        }
        batch = transfers.Batch<T>();
        DirtyIndex = transfers.NextCollection();
        Offset = batch.Add(this);
    }

    /// <summary>Returns a reference to a nonempty array's first physical element.</summary>
    /// <param name="array">The host array.</param>
    /// <returns>The first physical element.</returns>
    private static ref T First(Array array) => ref firstElements.GetOrAdd(array.GetType(), CreateFirstElement)(array);

    /// <summary>Builds a rank-specific CLR array address getter without per-element boxing.</summary>
    /// <param name="arrayType">The concrete CLR array type.</param>
    /// <returns>The cached host address getter.</returns>
    /// <remarks>The runtime stores rectangular elements contiguously in row-major order.
    /// Empty arrays never call this getter. Nonzero lower bounds are loaded for every dimension.</remarks>
    private static FirstElement CreateFirstElement(Type arrayType)
    {
        var method = new DynamicMethod("ArrayFirst", typeof(T).MakeByRefType(), [typeof(Array)],
            typeof(CapturedCollection<T>), true);
        var il = method.GetILGenerator();
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Castclass, arrayType);
        var lowerBound = typeof(Array).GetMethod(nameof(Array.GetLowerBound), [typeof(int)])!;
        for (int dimension = 0; dimension < arrayType.GetArrayRank(); dimension++)
        {
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, dimension);
            il.Emit(OpCodes.Callvirt, lowerBound);
        }
        il.Emit(OpCodes.Call, arrayType.GetMethod("Address", BindingFlags.Public | BindingFlags.Instance)!);
        il.Emit(OpCodes.Ret);
        return (FirstElement)method.CreateDelegate(typeof(FirstElement));
    }

    /// <inheritdoc/>
    internal override DeviceCollection View => new(
        batch.View.SubView(Snapshot.Length == 0 ? 0 : (long)Offset * batch.ElementSize,
            (long)Snapshot.Length * batch.ElementSize), Snapshot.Length,
        elementSize: batch.ElementSize, writes: transfers.Writes, dirtyIndex: DirtyIndex,
        shapeMetadata: transfers.Shapes, shapeOffset: shapeOffset, rank: rank, identity: Identity, context: transfers.Context);

    /// <summary>Copies the downloaded elements into the original collection.</summary>
    /// <param name="values">The downloaded leaf slice.</param>
    /// <exception cref="InvalidOperationException">A list changed length during execution.</exception>
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
            case Array array when array.Length > 0:
                values.CopyTo(MemoryMarshal.CreateSpan(ref First(array), array.Length));
                break;
        }
    }
}
