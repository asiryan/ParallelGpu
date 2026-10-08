using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace ParallelGpu;

/// <summary>Registers host-only snapshots for custom indexed collections used in ordinary GPU loops.</summary>
/// <remarks>Registration must precede loop compilation. Each collection type can be registered once.
/// GPU bodies can use its integer indexer and Count or Length property; adapter callbacks run on the host.</remarks>
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
    /// <exception cref="InvalidOperationException">The type is already registered or is a built-in collection.</exception>
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
        if (!type.GetProperties().Any(property => property.PropertyType == typeof(T) &&
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

    internal static bool SupportsRuntime(Type type) => BuiltinElement(type) is not null ||
        adapters.Keys.Any(candidate => candidate.IsAssignableFrom(type));

    private static Type? BuiltinElement(Type type)
    {
        if (!type.IsGenericType)
            return null;
        var definition = type.GetGenericTypeDefinition();
        return definition == typeof(ArraySegment<>) || definition == typeof(Memory<>) ||
            definition == typeof(ReadOnlyMemory<>) || definition == typeof(Span<>) || definition == typeof(ReadOnlySpan<>)
            ? type.GetGenericArguments()[0] : null;
    }

    internal static bool TryCapture<T>(object source, Dictionary<object, object> snapshots, out GpuCollectionSnapshot<T> snapshot)
    {
        switch (source)
        {
            case ArraySegment<T> segment:
                snapshot = new(segment.Array ?? Array.Empty<T>(), segment.Offset, segment.Count);
                break;
            case Memory<T> memory:
                snapshot = CaptureMemory((ReadOnlyMemory<T>)memory, false, snapshots);
                break;
            case ReadOnlyMemory<T> memory:
                snapshot = CaptureMemory(memory, true, snapshots);
                break;
            default:
                var adapter = FindAdapter(source.GetType(), typeof(T));
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
                if (snapshot.Identity is T[] native)
                {
                    if (native.Length != snapshot.Buffer.Length)
                        throw new InvalidOperationException("An array storage identity must have the same full length as its GPU snapshot.");
                    snapshot = new(native, snapshot.Offset, snapshot.Count, snapshot.IsReadOnly, snapshot.Commit, native);
                }
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

    private static Adapter? FindAdapter(Type runtime, Type element)
    {
        if (adapters.TryGetValue(runtime, out var direct))
            return direct;
        var candidates = adapters.Where(pair => pair.Value.Element == element && pair.Key.IsAssignableFrom(runtime)).ToArray();
        var mostSpecific = candidates.Where(candidate => !candidates.Any(other =>
            other.Key != candidate.Key && candidate.Key.IsAssignableFrom(other.Key))).ToArray();
        if (mostSpecific.Length > 1)
            throw new NotSupportedException($"Runtime collection '{runtime}' matches several unrelated GPU adapters; register its concrete type.");
        return mostSpecific.Length == 1 ? mostSpecific[0].Value : null;
    }
    private static GpuCollectionSnapshot<T> CaptureMemory<T>(ReadOnlyMemory<T> memory, bool readOnly,
        Dictionary<object, object> snapshots)
    {
        if (MemoryMarshal.TryGetArray(memory, out ArraySegment<T> segment))
            return new(segment.Array ?? Array.Empty<T>(), segment.Offset, segment.Count, readOnly);
        if (MemoryMarshal.TryGetMemoryManager<T, MemoryManager<T>>(memory, out var manager, out int start, out int length))
        {
            var storage = manager!.GetSpan();
            if (!snapshots.TryGetValue(manager, out var previous))
            {
                previous = new GpuCollectionSnapshot<T>(storage.ToArray(),
                    commit: values => values.AsSpan().CopyTo(manager.GetSpan()), identity: manager);
                snapshots.Add(manager, previous);
            }
            if (previous is not GpuCollectionSnapshot<T> shared || shared.Buffer.Length != storage.Length)
                throw new InvalidOperationException("Snapshots sharing a storage identity must use the same element type and full buffer length.");
            if (shared.Commit is null)
            {
                shared = new(shared.Buffer, commit: values => values.AsSpan().CopyTo(manager.GetSpan()), identity: manager);
                snapshots[manager] = shared;
            }
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
