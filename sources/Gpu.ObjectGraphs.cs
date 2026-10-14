namespace ParallelGpu;

public sealed partial class Gpu
{
    /// <summary>Uploads an explicitly adapted host object graph into bounded device records and references.</summary>
    /// <typeparam name="TObject">The managed host object type.</typeparam>
    /// <typeparam name="TRecord">The public unmanaged, ILGPU-supported payload type.</typeparam>
    /// <param name="roots">The ordered graph roots, including null entries.</param>
    /// <param name="adapter">The host payload and reference adapter.</param>
    /// <param name="capacity">The maximum record count, or -1 to use the existing object count.</param>
    /// <param name="referenceCapacity">The maximum outgoing-reference count, or -1 to use the existing count.</param>
    /// <returns>A reusable graph whose device view preserves null, shared references, and cycles.</returns>
    /// <remarks>Reserve extra record and reference capacity to allow device-side object creation.
    /// Payloads use their native ILGPU representation. Host objects are updated, and new host instances
    /// created, only when the returned graph's CopyToHost method is called.</remarks>
    /// <exception cref="ArgumentNullException">The roots or adapter is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A capacity is below -1 or too small for the existing graph.</exception>
    /// <exception cref="ArgumentException">The record type is not publicly visible.</exception>
    /// <exception cref="InvalidOperationException">The adapter returns null reference lists or the graph is too large.</exception>
    /// <exception cref="NotSupportedException">The record contains inline arrays or fixed-buffer storage.</exception>
    /// <exception cref="ObjectDisposedException">The executor has been disposed.</exception>
    public GpuObjectGraph<TObject, TRecord> UploadObjectGraph<TObject, TRecord>(
        IReadOnlyList<TObject?> roots, IGpuObjectGraphAdapter<TObject, TRecord> adapter,
        int capacity = -1, int referenceCapacity = -1)
        where TObject : class where TRecord : unmanaged
    {
        if (roots is null)
            throw new ArgumentNullException(nameof(roots));
        if (adapter is null)
            throw new ArgumentNullException(nameof(adapter));
        if (capacity < -1)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        if (referenceCapacity < -1)
            throw new ArgumentOutOfRangeException(nameof(referenceCapacity));
        lock (SyncRoot)
        {
            ThrowIfDisposed();
            ValidateVisible(typeof(TRecord));
            DeviceValueType.ValidateStorage(typeof(TRecord));
            var objects = new List<TObject>();
            var identities = new Dictionary<object, int>(ReferenceIdentityComparer.Instance);
            int Register(TObject? value)
            {
                if (value is null)
                    return 0;
                if (!identities.TryGetValue(value, out int index))
                {
                    if (objects.Count == int.MaxValue)
                        throw new InvalidOperationException("The host object graph has too many records.");
                    if (capacity >= 0 && objects.Count >= capacity)
                        throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity is too small for the existing objects.");
                    index = objects.Count;
                    objects.Add(value);
                    identities.Add(value, index);
                }
                return index + 1;
            }
            var rootIndices = roots.Select(Register).ToArray();
            var payloads = new List<TRecord>();
            var starts = new List<int>();
            var sizes = new List<int>();
            var targets = new List<int>();
            for (int i = 0; i < objects.Count; i++)
            {
                var source = objects[i];
                payloads.Add(adapter.ReadValue(source));
                var linked = adapter.ReadReferences(source) ??
                    throw new InvalidOperationException("The graph adapter must return a non-null outgoing-reference list.");
                int linkCount = linked.Count;
                if (linkCount < 0 || linkCount > int.MaxValue - targets.Count)
                    throw new InvalidOperationException("The host object graph has an invalid or excessive reference count.");
                if (referenceCapacity >= 0 && linkCount > referenceCapacity - targets.Count)
                    throw new ArgumentOutOfRangeException(nameof(referenceCapacity), "Reference capacity is too small for the existing graph.");
                starts.Add(targets.Count);
                sizes.Add(linkCount);
                for (int reference = 0; reference < linkCount; reference++)
                    targets.Add(Register(linked[reference]));
            }
            if (capacity == -1)
                capacity = objects.Count;
            if (referenceCapacity == -1)
                referenceCapacity = targets.Count;
            if (capacity < objects.Count)
                throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity is too small for the existing objects.");
            if (referenceCapacity < targets.Count)
                throw new ArgumentOutOfRangeException(nameof(referenceCapacity), "Reference capacity is too small for the existing graph.");
            int graphId = GpuDeviceIdentity.Allocate();
            GpuBuffer<TRecord>? valueBuffer = null;
            GpuBuffer<int>? offsetBuffer = null, lengthBuffer = null, referenceBuffer = null, rootBuffer = null, stateBuffer = null;
            try
            {
                valueBuffer = Allocate<TRecord>(capacity);
                offsetBuffer = Allocate<int>(capacity);
                lengthBuffer = Allocate<int>(capacity);
                referenceBuffer = Allocate<int>(referenceCapacity);
                rootBuffer = Upload(rootIndices);
                stateBuffer = Upload(new int[] { objects.Count, targets.Count, 0, 0 });
                valueBuffer.CopyFrom(0, payloads.ToArray());
                offsetBuffer.CopyFrom(0, starts.ToArray());
                lengthBuffer.CopyFrom(0, sizes.ToArray());
                referenceBuffer.CopyFrom(0, targets.ToArray());
                var graph = new GpuObjectGraph<TObject, TRecord>(this, graphId, adapter, objects, identities, valueBuffer,
                    offsetBuffer, lengthBuffer, referenceBuffer, rootBuffer, stateBuffer);
                buffers.Add(graph);
                return graph;
            }
            catch
            {
                stateBuffer?.Dispose();
                rootBuffer?.Dispose();
                referenceBuffer?.Dispose();
                lengthBuffer?.Dispose();
                offsetBuffer?.Dispose();
                valueBuffer?.Dispose();
                throw;
            }
        }
    }
}
