# Parallel.Gpu

Synchronous C# parallel loops on CUDA or OpenCL, powered by ILGPU. Ordinary lambdas use familiar arrays, lists, and memory views; the library compiles the body, transfers captured data, and copies written collections back before returning. Reuse a `Gpu` instance to reuse compiled kernels.

The library targets .NET Standard 2.1. The solution, samples, and tests use the .NET 8 SDK; GPU execution requires the corresponding installed driver/runtime. Reference `src/Parallel.Gpu.csproj` from your application.

## A first loop

```csharp
using ParallelGpu;

using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
int[] input = { 1, 2, 3 };
int[] output = new int[input.Length];
gpu.For(0, input.Length, i => output[i] = input[i] * 2);
// output: 2, 4, 6
```

`Auto` tries CUDA first, then OpenCL. It uses the CPU only when `AllowCpuFallback = true`. Choose `GpuBackend.Cuda`, `OpenCL`, or `Cpu` explicitly when needed; explicit GPU selections never fall back to the CPU. Inspect `gpu.Backend`, `gpu.DeviceName`, and `gpu.IsGpu` to see the selected accelerator.

`GpuParallel.For(...)` provides the same lambda API. Calls without options share a default executor; calls with `GpuOptions` create and dispose an executor for that call.

A loop covers `[fromInclusive, toExclusive)`. Empty and reversed ranges execute no body; a nonempty range can contain at most `Int32.MaxValue` iterations. Iterations run in unspecified order. They must access independent output locations unless an explicit atomic operation or reduction supplies synchronization.

## Values and collection shapes

Automatic lambda loops support:

| Values | Details |
| --- | --- |
| `byte`, `sbyte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong` | Standard integer types |
| `float`, `double`, `bool`, `char` | Full UTF-16 character values are preserved |
| `enum` | Any standard integer underlying type |
| `struct` | Recursively supported fields, including private, readonly, nested, and generic structures |
| `Nullable<T>` | Supported payload type; presence and payload are transferred separately |

Structures must contain no managed references and cannot use explicit layout, native pointers, or `ref struct` storage. Constructors, operators, fields, and helper methods are supported only when their executed code is device-compatible. Recognizing a value type does not make all its framework methods GPU-compatible. Examples include `Complex`, value tuples, and structures containing nullable fields.

Supported collection forms are rectangular arrays of any CLR rank, recursively nested arrays of arbitrary depth, and `List<T>` (also accessed through `IList<T>` or `IReadOnlyList<T>`). Nested arrays may combine rectangular containers and rectangular leaves. Array shape, nonzero lower bounds, empty dimensions, null branches, shared containers, and shared leaves are retained. Length, rank, dimension lengths, lower bounds, and upper bounds are available in the body. Existing elements can be read, written, and passed by reference; array topology stays fixed during an automatic loop.

```csharp
int[,,] volume = new int[2, 3, 4];
gpu.For<int>(0, volume.GetLength(0), x =>
{
    for (int y = 0; y < volume.GetLength(1); y++)
        for (int z = 0; z < volume.GetLength(2); z++)
            volume[x, y, z] = x * 100 + y * 10 + z;
});

int[] leaf = { 2, 4 };
int[][][] tree = { new int[][] { leaf, null!, leaf } };
gpu.For<int>(0, leaf.Length, i => tree[0][2][i] += 3);
// leaf: 5, 7; both tree leaves still reference the same array

int?[] nullable = { null, 2, 3 };
int? offset = 4;
gpu.For<int?>(0, nullable.Length, i => nullable[i] += offset);
// nullable: null, 6, 7
```

The optional `For<T>` overload requires every automatically transferred collection to have the same ultimate element type `T`. Use `For(...)` for mixed element types. Scalar captures are refreshed on each call and are read-only in the device body; copy a captured structure into a local before mutating it. Collection element writes and native device-view operations remain writable.

## Helpers and memory slices

Device-compatible helpers can receive and return supported collections, including multidimensional arrays, lists, spans, and references to elements. Transitive and generic helper calls use the same descriptor representation as captured collections.

```csharp
static void Transform(int[] input, IList<int> output, int index) =>
    output[index] = input[index] * 2;

var result = new List<int> { 0, 0, 0 };
gpu.For(0, input.Length, i => Transform(input, result, i));
```

`ArraySegment<T>`, `Memory<T>`, and `ReadOnlyMemory<T>` can be captured. `Span<T>` and `ReadOnlySpan<T>` can be locals and helper parameters. Integer indexers, `Count`/`Length`, `IsEmpty` where provided, `Span`, integer `Slice` overloads, array `AsSpan`/`AsMemory`, supported array-backed constructors, and implicit memory/span conversions are translated. Related views share underlying storage, including independently captured backing arrays. Read-only views reject writes. `Index`/`Range` slicing and other arbitrary collection methods are currently unsupported.

```csharp
int[] storage = { 1, 2, 3, 4 };
Memory<int> writable = storage.AsMemory(1, 2);
ReadOnlyMemory<int> readable = storage.AsMemory(1, 2);
int[] observed = new int[2];
gpu.For<int>(0, writable.Length, i =>
{
    writable.Span[i] += 10;
    observed[i] = readable.Span[i];
});
// storage: 1, 12, 13, 4; observed: 12, 13
```

Array-backed memory transfers its shared array. Memory-manager slices share a staged full-owner snapshot and commit; string-backed `ReadOnlyMemory<char>` uses a read-only UTF-16 snapshot. Aliasing between a custom memory manager and unrelated captures of its hidden storage requires that manager to expose its backing array through `TryGetArray`.

## Custom indexed collections

Register a host snapshot callback before any GPU compilation inspects the collection type:

```csharp
GpuCollectionAdapters.Register<IndexedSlice, int>(collection =>
    new GpuCollectionSnapshot<int>(collection.Buffer, collection.Offset, collection.Count));

public sealed class IndexedSlice
{
    public int[] Buffer { get; }
    public int Offset { get; }
    public int Count { get; }
    public IndexedSlice(int[] buffer, int offset, int count) =>
        (Buffer, Offset, Count) = (buffer, offset, count);
    public int this[int index]
    {
        get => Buffer[Offset + index];
        set => Buffer[Offset + index] = value;
    }
}
```

A registered collection exposes a readable integer indexer with the registered element type. The GPU uses the snapshot's indexer and `Count`/`Length`; the original host indexer is not called. Registration is permanent for that type; duplicate or late registrations are rejected. Interface/base adapters resolve the most specific matching runtime adapter; ambiguous unrelated adapters require a concrete-type registration.

`GpuCollectionSnapshot<T>` contains the full `Buffer`, slice `Offset`/`Count`, optional `isReadOnly`, optional shared `identity`, and optional `Action<T[]> commit`. Return a shared backing array to preserve aliases. For staged data, a shared identity reuses one full snapshot and one commit per invocation. An array identity denotes that exact native storage and must match the full buffer length. Commit receives the updated whole buffer after successful GPU execution and host downloads. Adapter callback exceptions can leave partial host changes; callbacks execute on the host and are responsible for their own side effects.

## Reductions and atomics

Use a reduction for a result shared by all iterations:

```csharp
int total = gpu.Reduce(new int[] { 1, 2, 3 }, new Int32Sum());
// total: 6

using var uploaded = gpu.Upload(new int[] { 1, 2, 3 });
int deviceTotal = gpu.Reduce(uploaded, new Int32Sum());
```

Built-in sums cover `int`, `long`, `uint`, `ulong`, `float`, and `double`. Integer sums use unchecked overflow. Custom public value types implement `IGpuReduction<T>` with pure device-compatible `Combine` and a two-sided `Identity`. The operation must be associative; input order is preserved, so commutativity is unnecessary. Empty input returns `Identity`. Floating-point rounding follows the parallel tree and may differ from sequential execution or between devices. Reduction input is unchanged.

`GpuAtomic` supplies `Add`, `Min`, `Max`, `Exchange`, and `CompareExchange` for the same six numeric types, plus integer `And`/`Or`/`Xor`. Operations return the previous value; compare-exchange arguments match `Interlocked.CompareExchange(ref target, value, comparand)`.

```csharp
int[] counter = { 0 };
gpu.For<int>(0, 1000, i => GpuAtomic.Add(ref counter[0], 1));
// counter[0]: 1000
```

## Bounded append output

Append into storage reserved on the host instead of calling `List.Add` in a device body:

```csharp
using var appended = gpu.CreateAppendBuffer<int>(input.Length);
var appendView = appended.View;
gpu.For(0, input.Length, i =>
{
    if (input[i] % 2 != 0)
        appendView.Add(input[i]);
});
int[] selected = appended.ToArray();
```

Concurrent appends reserve distinct slots; output order is unspecified. `TryAdd(value, out index)` reports the reserved slot or `-1`. Capacity is fixed, count saturates at capacity, and any exhausted reservation sets `Overflowed`; even `TryAdd` failure marks overflow. `ToArray()` rejects overflowed results with `InvalidOperationException`. Call `Reset()` before reuse. Reservation does not publish a completed value to other work items; consume produced output after the kernel completes.

`gpu.Collect<T,TKernel>(from, to, capacity, kernel)` provides a one-call alternative using a public value type implementing `IGpuCollectKernel<T>`. A capacity overflow rejects its output before values are returned.

## Explicit object graphs

Represent managed classes, polymorphic data, and object creation through an explicit record adapter. Reference-bearing structures require a host wrapper or a manually encoded record-and-links representation. `UploadObjectGraph<TObject,TRecord>` traverses host references iteratively, preserving reference identity, nulls, sharing, and cycles. `TObject` is a host class; `TRecord` is a public unmanaged GPU payload. Encode values and optional type tags in the record and expose ordered links through `IGpuObjectGraphAdapter<TObject,TRecord>`.

```csharp
var root = new Node(7) { Links = new Node?[] { null } };
using var graph = gpu.UploadObjectGraph<Node, NodeRecord>(
    new Node?[] { root }, new NodeAdapter(), capacity: 4, referenceCapacity: 8);
var graphView = graph.View;
gpu.For(0, 1, i =>
{
    var parent = graphView.GetRoot(0);
    var child = graphView.Create(new NodeRecord(12), referenceCount: 1);
    graphView.SetReference(child, 0, parent);
    graphView.SetReference(parent, 0, child);
});
var snapshot = graph.Download();       // Leaves root unchanged.
Node?[] updatedRoots = graph.CopyToHost(); // Creates child and restores the cycle.

public readonly record struct NodeRecord(int Value);
public sealed class Node
{
    public int Value;
    public Node?[] Links = Array.Empty<Node?>();
    public Node(int value) => Value = value;
}
public sealed class NodeAdapter : IGpuObjectGraphAdapter<Node, NodeRecord>
{
    public NodeRecord ReadValue(Node source) => new(source.Value);
    public IReadOnlyList<Node?> ReadReferences(Node source) => source.Links;
    public Node CreateObject(NodeRecord value) => new(value.Value);
    public void WriteValue(Node target, NodeRecord value) => target.Value = value.Value;
    public void WriteReferences(Node target, IReadOnlyList<Node?> references) =>
        target.Links = references.ToArray();
}
```

All adapter methods run on the host. Existing objects retain their identities; new records require distinct nonnull objects from `CreateObject`. `CopyToHost()` returns current roots without replacing the input root list. Downloaded snapshots remain usable after graph disposal; live handles are graph-specific and default handles mean null.

Record capacity, total reference-slot capacity, and root count are fixed on upload. Omitted capacities fit the existing graph exactly; reserve extra capacity for creation. Each record's outgoing-reference count is fixed, but links and roots can be replaced. `Create`/`TryCreate` reserve bounded records and links atomically. Newly created records must be read by other work items only in a later kernel after creation has completed; concurrent edits to the same existing payload or link still need caller synchronization.

Invalid/null/foreign graph access and capacity exhaustion set graph error flags. `Download()` and `CopyToHost()` reject the graph before reconstruction/write callbacks. Dispose a failed graph and upload a new one; the failed device graph is not restored. Adapter callback failures can partially change host objects. An ordinary lambda's successful return does not itself validate graph or append result flags: call the corresponding download method.

## Execution, errors, and lifetimes

Ordinary lambda compilation rejects unsupported operations rather than running the body on the CPU. Scalar captures are reread each invocation; collection snapshots are invocation-local. Aliases share device storage. Only written leaves, or leaves exposed by writable reference, are downloaded. Calls on one executor are serialized; do not mutate transferred host collections concurrently.

Ordinary loops defer nullable `Value` on null, null collection dereferences, invalid indices, invalid slices, and writes through read-only snapshots to host exceptions. These errors are checked before automatically transferred collections are copied back or snapshot commits run. This protects those host collections from partial kernel results. It does not roll back separately owned device buffers, append buffers, object graphs, or adapter callback side effects. Explicit value kernels use ILGPU's own execution contracts rather than the automatic lambda error transport.

`GpuBuffer<T>`, `GpuAppendBuffer<T>`, and `GpuObjectGraph<TObject,TRecord>` are disposable and owned by their executor; disposing the executor releases outstanding allocations. Borrowed device views must remain within their owner's lifetime and must be used with that executor. Ordinary lambda captures of native append/graph views validate owner and lifetime before launch. Their payload must be a public supported value type whose layout needs no automatic conversion; use `ushort` for characters and `int` for Boolean record fields. Mapped payloads are rejected before device compilation. With explicit `IGpuForKernel`/`IGpuCollectKernel` state, the caller must preserve view ownership and lifetime.

The automatic lambda path maps private structures, characters, booleans, and nullable values into device representations. Direct `Upload<T>`, `Allocate<T>`, reductions, append buffers, graph records, and explicit value kernels use public unmanaged ILGPU-compatible representations; they do not perform those automatic host-value conversions.

Intentional CLR limitations include arbitrary class instances or reference-bearing structures as direct loop values, strings as direct values, boxing, runtime reflection, managed allocation, `new` class objects, exception handlers/throwing bodies, delegates, dynamic/virtual managed dispatch, recursion, native pointers, and changing array/list topology inside automatic loops. Use snapshots, graph records, bounded append buffers, and reductions for their explicit GPU equivalents. Unsupported helper IL is rejected during translation.

## Build, samples, and verification

```powershell
dotnet build Parallel.Gpu.sln
dotnet test tests/Parallel.Gpu.Tests.csproj
$env:PARALLEL_GPU_TEST_BACKEND = 'Cuda'
dotnet test tests/Parallel.Gpu.Tests.csproj
dotnet run --project samples/FloatMatrixMultiplication
dotnet run --project samples/ComplexVectorMultiplication
```

Tests default to the ILGPU CPU backend for correctness checks. Set `PARALLEL_GPU_TEST_BACKEND` to `Cuda`, `OpenCL`, or `Auto` to require the selected GPU path. Tests cover array rank and nesting, lower bounds, aliases, nullable values, structures, helper translation, memory adapters, deferred errors, reductions, atomics, bounded output, object graphs, and disposal.
