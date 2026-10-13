namespace ParallelGpu;

/// <summary>Owns a bounded device snapshot of a host object graph and its host adapter.</summary>
/// <typeparam name="TObject">The managed host object type.</typeparam>
/// <typeparam name="TRecord">The unmanaged, ILGPU-supported payload type.</typeparam>
/// <remarks>Transfer traverses references iteratively and preserves object identity, null, sharing,
/// and cycles. Download creates an immutable value snapshot. CopyToHost explicitly applies changes
/// to original objects and creates host instances for new records. Adapter callbacks run on the host.
/// Capacity exhaustion or invalid device access rejects the graph before adapter write callbacks run;
/// dispose and upload a new graph after such a failure. Callback exceptions can leave partial host edits.</remarks>
public sealed class GpuObjectGraph<TObject, TRecord> : IDisposable
    where TObject : class where TRecord : unmanaged
{
    /// <summary>The owning executor.</summary>
    private readonly Gpu owner;
    /// <summary>The lifetime shared by all copied device views.</summary>
    private readonly GpuViewLifetime lifetime;
    /// <summary>The globally unique graph identity.</summary>
    private readonly int graphId;
    /// <summary>The host conversion and reconstruction contract.</summary>
    private readonly IGpuObjectGraphAdapter<TObject, TRecord> adapter;
    /// <summary>Stable host objects in device-record order.</summary>
    private readonly List<TObject> objects;
    /// <summary>Existing host objects mapped by reference identity.</summary>
    private readonly Dictionary<object, int> identities;
    /// <summary>The device records.</summary>
    private readonly GpuBuffer<TRecord> values;
    /// <summary>The outgoing-reference offset of each record.</summary>
    private readonly GpuBuffer<int> offsets;
    /// <summary>The outgoing-reference count of each record.</summary>
    private readonly GpuBuffer<int> lengths;
    /// <summary>The bounded outgoing-reference slots.</summary>
    private readonly GpuBuffer<int> references;
    /// <summary>The fixed root slots.</summary>
    private readonly GpuBuffer<int> roots;
    /// <summary>The allocation counts and error flags.</summary>
    private readonly GpuBuffer<int> state;
    /// <summary>Whether this wrapper has been disposed.</summary>
    private bool disposed;
    /// <summary>Whether adapter callbacks are applying a snapshot to this graph.</summary>
    private bool copyToHostInProgress;

    /// <summary>Takes ownership of initialized graph allocations.</summary>
    /// <param name="owner">The executor.</param>
    /// <param name="graphId">The unique graph identity.</param>
    /// <param name="adapter">The host object adapter.</param>
    /// <param name="objects">The original host objects.</param>
    /// <param name="identities">The reference-identity index.</param>
    /// <param name="values">The record pool.</param>
    /// <param name="offsets">The per-record reference offsets.</param>
    /// <param name="lengths">The per-record reference counts.</param>
    /// <param name="references">The reference pool.</param>
    /// <param name="roots">The root slots.</param>
    /// <param name="state">The allocation counts and error flags.</param>
    internal GpuObjectGraph(Gpu owner, int graphId, IGpuObjectGraphAdapter<TObject, TRecord> adapter,
        List<TObject> objects, Dictionary<object, int> identities, GpuBuffer<TRecord> values,
        GpuBuffer<int> offsets, GpuBuffer<int> lengths, GpuBuffer<int> references,
        GpuBuffer<int> roots, GpuBuffer<int> state)
    {
        this.owner = owner;
        lifetime = new(owner);
        this.graphId = graphId;
        this.adapter = adapter;
        this.objects = objects;
        this.identities = identities;
        this.values = values;
        this.offsets = offsets;
        this.lengths = lengths;
        this.references = references;
        this.roots = roots;
        this.state = state;
    }

    /// <summary>Gets the maximum number of object records.</summary>
    public int Capacity => values.Length;

    /// <summary>Gets the maximum total number of outgoing references.</summary>
    public int ReferenceCapacity => references.Length;

    /// <summary>Gets the fixed number of graph roots.</summary>
    public int RootCount => roots.Length;

    /// <summary>Gets a device view for kernels executed by this graph's owner.</summary>
    /// <remarks>The view does not own memory and must not outlive the graph or executor.</remarks>
    /// <exception cref="ObjectDisposedException">The graph or executor has been disposed.</exception>
    public GpuObjectGraphView<TRecord> View
    {
        get
        {
            lock (owner.SyncRoot)
            {
                ThrowIfDisposed();
                return new(lifetime.Id, graphId, values.View, offsets.View, lengths.View, references.View, roots.View, state.View);
            }
        }
    }

    /// <summary>Gets the stable device reference of an original or reconstructed host object.</summary>
    /// <param name="value">The host object, or null.</param>
    /// <returns>The object's graph reference, or null.</returns>
    /// <exception cref="ArgumentException">The host object is not part of this graph.</exception>
    /// <exception cref="ObjectDisposedException">The graph or executor has been disposed.</exception>
    public GpuObjectHandle GetHandle(TObject? value)
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (value is null)
                return default;
            if (!identities.TryGetValue(value, out int index))
                throw new ArgumentException("The host object is not part of this graph.", nameof(value));
            return new(graphId, index + 1);
        }
    }

    /// <summary>Downloads validated values and topology without modifying any host objects.</summary>
    /// <returns>An immutable graph snapshot that remains usable after graph disposal.</returns>
    /// <exception cref="InvalidOperationException">Record or reference capacity was exhausted, or graph access was invalid.</exception>
    /// <exception cref="ObjectDisposedException">The graph or executor has been disposed.</exception>
    public GpuObjectGraphSnapshot<TRecord> Download()
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            var status = state.ToArray();
            if (status[2] != 0)
                throw new InvalidOperationException("GPU object graph exceeded its record or reference capacity. Dispose it and upload a graph with sufficient capacity.");
            if (status[3] != 0)
                throw new InvalidOperationException("GPU object graph encountered a null, foreign, or invalid object/reference access.");
            int count = status[0];
            int referenceCount = status[1];
            if (count < 0 || count > Capacity || referenceCount < 0 || referenceCount > ReferenceCapacity)
                throw new InvalidOperationException("GPU object graph allocation metadata is invalid.");
            var payloads = new TRecord[count];
            var starts = new int[count];
            var sizes = new int[count];
            var targets = new int[referenceCount];
            values.CopyTo(0, payloads);
            offsets.CopyTo(0, starts);
            lengths.CopyTo(0, sizes);
            references.CopyTo(0, targets);
            var rootIndices = roots.ToArray();
            var links = new GpuObjectHandle[count][];
            for (int i = 0; i < count; i++)
            {
                if (starts[i] < 0 || sizes[i] < 0 || starts[i] > referenceCount || sizes[i] > referenceCount - starts[i])
                    throw new InvalidOperationException("GPU object graph reference metadata is invalid.");
                links[i] = new GpuObjectHandle[sizes[i]];
                for (int j = 0; j < sizes[i]; j++)
                    links[i][j] = ConvertTarget(targets[starts[i] + j], count);
            }
            var rootHandles = rootIndices.Select(index => ConvertTarget(index, count)).ToArray();
            return new(graphId, payloads, links, rootHandles);
        }
    }

    /// <summary>Applies a validated device graph to host objects while preserving stable identities.</summary>
    /// <returns>The current host roots; newly selected roots are returned rather than replacing the original input list.</returns>
    /// <remarks>All new host instances are created before payload and reference writes. Existing instances
    /// are reused, including shared and cyclic references. Adapter-created objects must be distinct.
    /// Capacity and graph-access errors are checked before any adapter callback. Adapter exceptions
    /// may leave partially applied host changes. Adapter callbacks cannot recursively call
    /// CopyToHost on this graph; read-only graph operations remain available.</remarks>
    /// <exception cref="InvalidOperationException">The graph is invalid, the adapter returns null or duplicate
    /// instances, or an adapter callback recursively calls CopyToHost on this graph.</exception>
    /// <exception cref="ObjectDisposedException">The graph or executor has been disposed.</exception>
    public TObject?[] CopyToHost()
    {
        lock (owner.SyncRoot)
        {
            ThrowIfDisposed();
            if (copyToHostInProgress)
                throw new InvalidOperationException("Adapter callbacks cannot recursively call CopyToHost on the same graph.");
            copyToHostInProgress = true;
            try
            {
                var snapshot = Download();
                var instances = objects.ToList();
                var seen = new HashSet<object>(instances, ReferenceIdentityComparer.Instance);
                for (int i = instances.Count; i < snapshot.Data.Length; i++)
                {
                    var created = adapter.CreateObject(snapshot.Data[i]);
                    if (created is null || !seen.Add(created))
                        throw new InvalidOperationException("The graph adapter must create a distinct non-null object for each new record.");
                    instances.Add(created);
                }
                // Retain each new instance before write callbacks so a retry preserves its identity.
                for (int i = objects.Count; i < instances.Count; i++)
                {
                    identities.Add(instances[i], i);
                    objects.Add(instances[i]);
                }
                for (int i = 0; i < instances.Count; i++)
                    adapter.WriteValue(instances[i], snapshot.Data[i]);
                for (int i = 0; i < instances.Count; i++)
                {
                    var linked = snapshot.Links[i].Select(handle => handle.IsNull ? null : instances[handle.Index]).ToArray();
                    adapter.WriteReferences(instances[i], Array.AsReadOnly(linked));
                }
                return snapshot.Roots.Select(handle => handle.IsNull ? null : instances[handle.Index]).ToArray();
            }
            finally
            {
                copyToHostInProgress = false;
            }
        }
    }

    /// <summary>Releases all graph allocations; repeated calls have no effect.</summary>
    public void Dispose()
    {
        lock (owner.SyncRoot)
        {
            if (disposed)
                return;
            lifetime.Dispose();
            state.Dispose();
            roots.Dispose();
            references.Dispose();
            lengths.Dispose();
            offsets.Dispose();
            values.Dispose();
            disposed = true;
            owner.Release(this);
        }
    }

    /// <summary>Validates a downloaded one-based target index.</summary>
    /// <param name="index">The downloaded target index.</param>
    /// <param name="count">The valid object count.</param>
    /// <returns>The nullable graph handle.</returns>
    private GpuObjectHandle ConvertTarget(int index, int count)
    {
        if (index < 0 || index > count)
            throw new InvalidOperationException("GPU object graph contains an invalid target reference.");
        return index == 0 ? default : new(graphId, index);
    }

    /// <summary>Rejects host operations after graph or owner disposal.</summary>
    private void ThrowIfDisposed()
    {
        if (disposed)
            throw new ObjectDisposedException(GetType().FullName);
        owner.ThrowIfDisposed();
    }
}

/// <summary>Allocates process-wide identities for graphs and owned device views.</summary>
internal static class GpuDeviceIdentity
{
    /// <summary>The most recently assigned device identity.</summary>
    private static int next;

    /// <summary>Allocates a unique positive identity for a graph or owned view.</summary>
    /// <returns>The new identity.</returns>
    internal static int Allocate()
    {
        int current = System.Threading.Volatile.Read(ref next);
        while (true)
        {
            if (current == int.MaxValue)
                throw new InvalidOperationException("GPU object graph identities have been exhausted.");
            int observed = System.Threading.Interlocked.CompareExchange(ref next, current + 1, current);
            if (observed == current)
                return current + 1;
            current = observed;
        }
    }
}
