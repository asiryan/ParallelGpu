using System.Buffers;
using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.InteropServices;

namespace ParallelGpu;

/// <summary>Registers host-only snapshots for custom indexed collections used in ordinary GPU loops.</summary>
/// <remarks>Register before GPU compilation first inspects the type. Each type can be registered once.
/// GPU bodies can use its integer indexer (including inherited or renamed indexers) and Count or Length
/// property. Device lengths come from the snapshot's Count; adapter callbacks run on the host.
/// Registered list subclasses use adapters when captured through their registered declaration;
/// List, IList, and IReadOnlyList declarations use native array or list storage. A registered
/// capture declaration selects its own adapter before runtime adapter resolution.</remarks>
public static class GpuCollectionAdapters
{
    private sealed record Adapter(Type Element, Func<object, object> Capture);
    private static readonly ConcurrentDictionary<Type, Adapter> adapters = new();
    private static readonly object registrationSync = new();
    private static readonly HashSet<Type> frozen = new();

    /// <summary>Registers a snapshot adapter for a custom collection type.</summary>
    /// <typeparam name="TCollection">The declared indexed collection type.</typeparam>
    /// <typeparam name="T">The supported element type returned by its integer indexer.</typeparam>
    /// <param name="capture">The host callback that exposes shared backing storage or stages a snapshot.</param>
    /// <exception cref="ArgumentNullException">The callback is null.</exception>
    /// <exception cref="ArgumentException">The collection type has no matching integer indexer.</exception>
    /// <exception cref="InvalidOperationException">The type is built-in, already registered, or already inspected for compilation.</exception>
    /// <exception cref="NotSupportedException">The element type is unsupported.</exception>
    public static void Register<TCollection, T>(Func<TCollection, GpuCollectionSnapshot<T>> capture)
    {
        if (capture is null)
            throw new ArgumentNullException(nameof(capture));
        if (!DeviceValueType.IsSupported(typeof(T)))
            throw new NotSupportedException($"'{typeof(T)}' is not a supported GPU collection element type.");
        var type = typeof(TCollection);
        if (type.IsArray || BuiltinElement(type) is not null ||
            (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(List<>) ||
                type.GetGenericTypeDefinition() == typeof(IList<>) || type.GetGenericTypeDefinition() == typeof(IReadOnlyList<>))))
            throw new InvalidOperationException($"Collection '{type}' already has a built-in GPU representation.");
        if (!Properties(type).Any(property => property.PropertyType == typeof(T) &&
            property.GetIndexParameters().Length == 1 && property.GetIndexParameters()[0].ParameterType == typeof(int) &&
            property.GetMethod is not null))
            throw new ArgumentException($"Collection '{type}' must expose a readable integer indexer of type '{typeof(T)}'.", nameof(capture));
        lock (registrationSync)
        {
            if (frozen.Contains(type))
                throw new InvalidOperationException($"Collection type '{type}' has already been inspected for GPU compilation. Register its adapter before compiling any loop using that type.");
            if (!adapters.TryAdd(type, new(typeof(T), value => capture((TCollection)value))))
                throw new InvalidOperationException($"A GPU collection adapter for '{type}' is already registered.");
        }
    }

    /// <summary>Resolves a built-in slice or registered element type, freezing custom type registration.</summary>
    internal static Type? ElementType(Type type)
    {
        var builtin = BuiltinElement(type);
        if (builtin is not null)
            return builtin;
        lock (registrationSync)
        {
            frozen.Add(type);
            return adapters.TryGetValue(type, out var adapter) ? adapter.Element : null;
        }
    }

    /// <summary>Checks whether a runtime type is a built-in slice or matches a registered adapter.</summary>
    internal static bool SupportsRuntime(Type type) => BuiltinElement(type) is not null ||
        adapters.Keys.Any(candidate => candidate.IsAssignableFrom(type));

    /// <summary>Finds public properties, including properties inherited by interfaces.</summary>
    private static IEnumerable<PropertyInfo> Properties(Type type) => type.IsInterface
        ? type.GetInterfaces().Append(type).SelectMany(candidate => candidate.GetProperties())
        : type.GetProperties();

    /// <summary>Resolves registered indexer accessors by metadata rather than their generated names.</summary>
    internal static string? AccessorOperation(Type type, MethodInfo method, Type element)
    {
        foreach (var property in Properties(type))
        {
            var indices = property.GetIndexParameters();
            if (property.PropertyType == element && indices.Length == 1 && indices[0].ParameterType == typeof(int))
            {
                if (SameAccessor(property.GetMethod, method))
                    return "get_Item";
                if (SameAccessor(property.SetMethod, method))
                    return "set_Item";
            }
            if (indices.Length == 0 && property.PropertyType == typeof(int) &&
                property.Name is "Count" or "Length" && SameAccessor(property.GetMethod, method))
                return "get_Count";
        }
        return null;
    }

    /// <summary>Matches inherited accessor metadata independently of its reflected receiver type.</summary>
    private static bool SameAccessor(MethodInfo? candidate, MethodInfo method)
    {
        if (candidate is null)
            return false;
        candidate = candidate.GetBaseDefinition();
        method = method.GetBaseDefinition();
        return candidate.Module == method.Module && candidate.MetadataToken == method.MetadataToken &&
            candidate.DeclaringType == method.DeclaringType;
    }

    /// <summary>Detects adapter upcasts that would keep storage for a different, hidden indexer.</summary>
    internal static bool ChangesIndexer(Type source, Type destination, Type element)
    {
        if (source == destination || !adapters.ContainsKey(source) || !adapters.ContainsKey(destination))
            return false;
        MethodInfo? Indexer(Type type)
        {
            var properties = Properties(type).Where(property => property.PropertyType == element &&
                property.GetIndexParameters() is [{ ParameterType: var index }] && index == typeof(int) &&
                property.GetMethod is not null).ToArray();
            return properties.FirstOrDefault(property => !properties.Any(other =>
                property.DeclaringType != other.DeclaringType &&
                property.DeclaringType!.IsAssignableFrom(other.DeclaringType)))?.GetMethod;
        }
        var sourceIndexer = Indexer(source);
        var destinationIndexer = Indexer(destination);
        if (!source.IsInterface && destinationIndexer?.DeclaringType is { IsInterface: true } contract &&
            contract.IsAssignableFrom(source))
        {
            // A concrete getter and the interface it implements have different metadata.
            var map = source.GetInterfaceMap(contract);
            int index = Array.FindIndex(map.InterfaceMethods, method => SameAccessor(method, destinationIndexer));
            if (index >= 0)
                destinationIndexer = map.TargetMethods[index];
        }
        return sourceIndexer is not null && destinationIndexer is not null &&
            !SameAccessor(sourceIndexer, destinationIndexer);
    }

    /// <summary>Finds the element type of built-in segment, memory, and span representations.</summary>
    private static Type? BuiltinElement(Type type)
    {
        if (!type.IsGenericType)
            return null;
        var definition = type.GetGenericTypeDefinition();
        return definition == typeof(ArraySegment<>) || definition == typeof(Memory<>) ||
            definition == typeof(ReadOnlyMemory<>) || definition == typeof(Span<>) || definition == typeof(ReadOnlySpan<>)
            ? type.GetGenericArguments()[0] : null;
    }

    /// <summary>Captures built-in slices or adapted storage while reusing buffers for shared storage identities.</summary>
    internal static bool TryCapture<T>(object source, Type? declaredType, Dictionary<object, object> snapshots,
        out GpuCollectionSnapshot<T> snapshot, out Action<T[]>? copyMemory)
    {
        copyMemory = null;
        switch (source)
        {
            case ArraySegment<T> segment:
                snapshot = new(segment.Array ?? Array.Empty<T>(), segment.Offset, segment.Count);
                break;
            case Memory<T> memory:
                snapshot = CaptureMemory((ReadOnlyMemory<T>)memory, false, snapshots, out copyMemory);
                break;
            case ReadOnlyMemory<T> memory:
                snapshot = CaptureMemory(memory, true, snapshots, out copyMemory);
                break;
            default:
                // Different declarations of one object can expose independent indexers and storage.
                var adapter = declaredType is not null && adapters.TryGetValue(declaredType, out var declared)
                    ? declared : FindAdapter(source.GetType(), typeof(T));
                if (adapter is null)
                {
                    snapshot = default;
                    return false;
                }
                if (adapter.Element != typeof(T))
                    throw new ArgumentException("The registered GPU adapter element type differs from the captured collection type.");
                snapshot = (GpuCollectionSnapshot<T>)adapter.Capture(source);
                if (snapshot.Buffer is null || snapshot.Identity is null)
                    throw new InvalidOperationException("A GPU collection adapter returned an uninitialized snapshot.");
                if (snapshot.Identity is Array nativeArray)
                {
                    if (nativeArray.GetType().GetElementType() != typeof(T) || nativeArray.Length != snapshot.Buffer.Length)
                        throw new InvalidOperationException("An array storage identity must have the same element type and full buffer length as its GPU snapshot.");
                    if (nativeArray is T[] native)
                        snapshot = new(native, snapshot.Offset, snapshot.Count, snapshot.IsReadOnly, snapshot.Commit, native);
                }
                else if (ListElement(snapshot.Identity.GetType()) is Type listElement &&
                    (listElement != typeof(T) || ((List<T>)snapshot.Identity).Count != snapshot.Buffer.Length))
                    throw new InvalidOperationException("A list storage identity must have the same element type and full buffer length as its GPU snapshot.");
                if (snapshots.TryGetValue(snapshot.Identity, out var previous))
                {
                    if (previous is not GpuCollectionSnapshot<T> shared || shared.Buffer.Length != snapshot.Buffer.Length)
                        throw new InvalidOperationException("Snapshots sharing a storage identity must use the same element type and full buffer length.");
                    snapshot = new(shared.Buffer, snapshot.Offset, snapshot.Count, snapshot.IsReadOnly,
                        shared.Commit ?? snapshot.Commit, snapshot.Identity);
                }
                else
                    snapshots.Add(snapshot.Identity, snapshot);
                break;
        }
        return true;
    }

    /// <summary>Finds the most specific assignable adapter with the requested element type, rejecting ambiguity.</summary>
    private static Adapter? FindAdapter(Type runtime, Type element)
    {
        if (adapters.TryGetValue(runtime, out var direct) && direct.Element == element)
            return direct;
        var candidates = adapters.Where(pair => pair.Value.Element == element && pair.Key.IsAssignableFrom(runtime)).ToArray();
        var mostSpecific = candidates.Where(candidate => !candidates.Any(other =>
            other.Key != candidate.Key && candidate.Key.IsAssignableFrom(other.Key))).ToArray();
        if (mostSpecific.Length > 1)
            throw new NotSupportedException($"Runtime collection '{runtime}' matches several unrelated GPU adapters; register its concrete type.");
        return mostSpecific.Length == 1 ? mostSpecific[0].Value : null;
    }

    /// <summary>Finds the native element type of a list storage identity, including derived lists.</summary>
    private static Type? ListElement(Type type)
    {
        for (Type? current = type; current is not null; current = current.BaseType)
            if (current.IsGenericType && current.GetGenericTypeDefinition() == typeof(List<>))
                return current.GetGenericArguments()[0];
        return null;
    }

    /// <summary>Shares array storage or stages a full memory owner while retaining slice offsets and read-only access.</summary>
    /// <remarks>Memory-manager copyback is separate from the first user callback for the shared identity.</remarks>
    private static GpuCollectionSnapshot<T> CaptureMemory<T>(ReadOnlyMemory<T> memory, bool readOnly,
        Dictionary<object, object> snapshots, out Action<T[]>? copyMemory)
    {
        copyMemory = null;
        if (MemoryMarshal.TryGetArray(memory, out ArraySegment<T> segment))
            return new(segment.Array ?? Array.Empty<T>(), segment.Offset, segment.Count, readOnly);
        // Snapshot the full owner once so overlapping slices share storage and a commit.
        if (MemoryMarshal.TryGetMemoryManager<T, MemoryManager<T>>(memory, out var manager, out int start, out int length))
        {
            var storage = manager!.GetSpan();
            if (!snapshots.TryGetValue(manager, out var previous))
            {
                previous = new GpuCollectionSnapshot<T>(storage.ToArray(), identity: manager);
                snapshots.Add(manager, previous);
            }
            if (previous is not GpuCollectionSnapshot<T> shared || shared.Buffer.Length != storage.Length)
                throw new InvalidOperationException("Snapshots sharing a storage identity must use the same element type and full buffer length.");
            copyMemory = values => values.AsSpan().CopyTo(manager.GetSpan());
            return new(shared.Buffer, start, length, readOnly, shared.Commit, manager);
        }
        if (typeof(T) == typeof(char) && MemoryMarshal.TryGetString((ReadOnlyMemory<char>)(object)memory,
            out string? text, out int stringStart, out int stringLength))
        {
            if (!snapshots.TryGetValue(text!, out var previous))
            {
                previous = new GpuCollectionSnapshot<T>((T[])(object)text!.ToCharArray(), true, identity: text);
                snapshots.Add(text!, previous);
            }
            if (previous is not GpuCollectionSnapshot<T> shared || shared.Buffer.Length != text!.Length)
                throw new InvalidOperationException("Snapshots sharing a storage identity must use the same element type and full buffer length.");
            return new(shared.Buffer, stringStart, stringLength, true, identity: text);
        }
        throw new NotSupportedException("The memory owner cannot expose GPU-compatible contiguous storage.");
    }
}
