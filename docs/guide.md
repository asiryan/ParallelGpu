# API guide

Start with the [quick start](../README.md#quick-start). The examples below reuse its `gpu` instance; memory views also require `using System;`. Numeric overloads are listed in the [math reference](math.md).

## Values and collections

The automatic `gpu.For(...)` lambda API maps these values to device storage:

| Value | Support |
| --- | --- |
| Integers | `byte`, `sbyte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong` |
| Other primitives | `float`, `double`, `bool`, `char` (full UTF-16 values) |
| Enums | Any standard integer underlying type |
| Structures | Supported fields recursively, including private, readonly, nested, and generic structures |
| `Nullable<T>` | Any supported payload type |

Structures cannot contain managed references, native pointers, explicit layout, inline arrays, or fixed buffers. Value tuples and `Complex` work when their executed methods are device-compatible; supporting a type does not imply support for every method on it.

| Collection or view | Support |
| --- | --- |
| Arrays of any rank | Dimensions and nonzero lower bounds are retained. |
| Jagged and mixed rectangular/jagged arrays | Null branches, empty dimensions, and shared references are retained. |
| `List<T>`, also via `IList<T>` or `IReadOnlyList<T>` | Indexed access with a fixed size during execution. |
| `ArraySegment<T>`, `Memory<T>`, `ReadOnlyMemory<T>` | Capturable slices; overlapping views share storage. |
| `Span<T>`, `ReadOnlySpan<T>` | Locals and helper parameters; integer indexing and slicing. |

Lists and contiguous views require supported value elements. Use jagged arrays to access nested arrays. `For<T>(...)` requires every automatically transferred collection to have ultimate element type `T`; use non-generic `For(...)` for mixed types. Device-compatible helpers may receive and return supported collections and references to elements.

Memory views support `Slice`, `Span`, `Length`, and `IsEmpty` where available; segments expose `Count`. Array `AsSpan`/`AsMemory` and supported constructors/conversions work in helpers. `Index`/`Range` slicing and arbitrary collection methods are unsupported. String-backed `ReadOnlyMemory<char>` is readable; custom memory managers use host snapshots and commits.

```csharp
int[] storage = { 1, 2, 3, 4 };
Memory<int> slice = storage.AsMemory(1, 2);
ReadOnlyMemory<int> readOnly = slice;
int[] observed = new int[2];

gpu.For(0, slice.Length, i =>
{
    slice.Span[i] += 10;
    observed[i] = readOnly.Span[i];
});
// storage: 1, 12, 13, 4; observed: 12, 13
```

## Native buffers and kernels

`Upload<T>(array)` copies data to a reusable device buffer; `Allocate<T>(length)` creates zero-filled storage. `DeviceView` exposes a backend-independent `GpuView<T>` with `Length` and a ref indexer. See the [buffer example](../README.md#keep-data-on-the-device).

| Transfer | Range |
| --- | --- |
| `CopyFrom(array)`, `CopyTo(array)` | Entire buffer; array length must match. |
| `CopyFrom(destinationOffset, source, sourceOffset, count)` | Array range to device. |
| `CopyTo(sourceOffset, destination, destinationOffset, count)` | Device range to array. |
| `CopyFrom(destinationOffset, source)`, `CopyTo(sourceOffset, destination)` | Entire array or span at the device offset. |

Transfers are synchronous. Span uploads accept `ReadOnlySpan<T>`; downloads accept `Span<T>`. Empty ranges at the buffer end are valid. Automatic collection slices transfer their full backing storage to preserve aliases; partial buffer copies transfer only the requested range.

Native buffers and explicit kernels use ILGPU's native type representation. They do not apply automatic mappings for private structures, `bool`, `char`, or nullable values. Kernel types, `ForEach`/`Reduce` element types, and payloads captured through buffer, append, or graph views must be public and ILGPU-compatible; use `ushort` for characters and `int` for Boolean fields.

For explicit kernels, implement [IGpuForKernel](../sources/IGpuForKernel.cs) or [IGpuForEachKernel](../sources/IGpuForEachKernel.cs) as a public value type. The explicit APIs follow ILGPU's execution contracts.

Views borrow memory from their owner. Dispose buffers, append buffers, graphs, and the executor when finished. Lambda captures check that views are alive and belong to the executor. Explicit kernels must enforce those requirements themselves. Native buffer indices must be in range in either API; device writes are not rolled back on failure. `buffer.View` exposes the underlying ILGPU view for interoperability.

## Reductions and atomics

```csharp
int total = gpu.Reduce(new int[] { 1, 2, 3 }, new Int32Sum());
// total: 6

int[] counter = { 0 };
gpu.For(0, 1000, i => GpuAtomic.Add(ref counter[0], 1));
// counter[0]: 1000
```

Built-in sums cover `int`, `long`, `uint`, `ulong`, `float`, and `double`. `Reduce` also accepts an uploaded buffer. Custom [IGpuReduction](../sources/IGpuReduction.cs) operations need a pure associative `Combine` and a two-sided `Identity`. Input order is preserved, but grouping is parallel. Empty input returns `Identity`; a single element is returned without combining. Integer sums use unchecked overflow; floating-point results may differ from sequential sums and between devices.

`GpuAtomic` provides `Add`, `Min`, `Max`, `Exchange`, and `CompareExchange` for those six types, plus integer `And`, `Or`, and `Xor`. Operations return the previous value. Compare-exchange uses the same argument order as `Interlocked.CompareExchange`.

## Bounded output

```csharp
int[] input = { 1, 2, 3 };
using var appended = gpu.CreateAppendBuffer<int>(input.Length);
var appendView = appended.View;

gpu.For(0, input.Length, i =>
{
    if (input[i] % 2 != 0)
        appendView.Add(input[i]);
});
int[] selected = appended.ToArray();
// selected contains 1 and 3 in unspecified order
```

Capacity is fixed. `TryAdd(value, out index)` reports success and the reserved slot; failure returns index `-1` and records overflow. `ToArray()` rejects overflowed output. Call `Reset()` before reuse, and read produced values only after the kernel completes. `Collect<T,TKernel>` is the equivalent one-call API for `IGpuCollectKernel<T>`.

## Collection adapters

Register `GpuCollectionAdapters.Register<TCollection,T>` once, before compiling any loop using that type. The host callback returns a `GpuCollectionSnapshot<T>` describing a backing array, slice, read-only status, optional shared identity, and commit callback. Integer indexers may be inherited or renamed; `Count`/`Length` comes from the snapshot. See [working examples](../tests/CollectionAdapterTests.cs).

Return the same backing array for shared storage. For staged snapshots, use a shared identity with the same element type and full buffer length. An array or list identity denotes that exact native storage. Only written storage is committed, at most once per identity after successful execution and download. A shared memory manager is updated before the user commit. Callback exceptions can leave partial host changes.

The captured declaration determines access:

- A registered declaration uses its adapter, slice, and read-only restrictions, including for a `List<T>` subclass.
- Capturing an adapted `List<T>` subclass as `List<T>`, `IList<T>`, or `IReadOnlyList<T>` uses its native list storage. Passing it to a native-list helper exposes the original full list; writes and commits follow the accessed storage.
- Runtime adapter resolution requires matching indexer and `Count` accessors. List subclasses that reimplement native interface accessors are rejected. Reference conversions that change the element type or select different indexer or length accessors are unsupported; capture the required declaration separately.

## Object graphs

Use `UploadObjectGraph<TObject,TRecord>` with an `IGpuObjectGraphAdapter<TObject,TRecord>` to map host classes to public unmanaged records and explicit links. Nulls, shared references, and cycles are preserved. See [working examples](../tests/ObjectGraphTests.cs).

| Operation | Purpose |
| --- | --- |
| Upload with `capacity` and `referenceCapacity` | Reserve room for records and outgoing links; defaults fit the existing graph. |
| `graph.View` | Read/write records, relink roots and references, and create records on the device. |
| `graph.Download()` | Validate device errors and return a value snapshot. |
| `graph.CopyToHost()` | Validate device errors, update original objects, and reconstruct new objects through the adapter. |

Root count, pool capacities, and each record's reference count are fixed. Newly created records can be shared with other iterations in a later kernel. Handles belong to their graph. Invalid access or capacity exhaustion makes subsequent downloads fail; dispose the graph and upload a new one.

Adapter callbacks run on the host. They cannot recursively call `CopyToHost()` on the same graph. Callback failures may leave partial host edits; already created host identities are retained for a retry. Append and graph errors require an explicit result download to validate them.

## Execution rules

- Loops cover `[fromInclusive, toExclusive)` with at most `Int32.MaxValue` iterations. Empty or reversed ranges execute no body. Iterations are unordered; avoid races with independent writes, atomics, or reductions.
- Scalar captures, including volatile fields, are read-only host snapshots refreshed per call. Collection elements may be writable, but sizes and array topology cannot change during execution.
- Calls on one `Gpu` are serialized. Do not modify transferred host collections concurrently. Static `GpuParallel` calls without options share an executor; nonempty calls with options create one per call.
- Automatic compilation requires an ordinary `void Method(int)` with inspectable IL and `Reflection.Emit`. Multicast delegates, closed static delegates with bound arguments, `DynamicMethod` delegates, and Native AOT are unsupported.
- Loop bodies and helpers cannot use managed allocation, reference-bearing values, boxing, reflection, exception handlers, checked integer arithmetic/conversions, dynamic managed dispatch, delegates, native pointers, or `ref struct` storage. Supported span locals and parameters are the exception to the `ref struct` restriction.

Automatic loops report bounds, null access, read-only writes, absent nullable `Value`, guarded arithmetic errors, and invalid supported API arguments as host exceptions before collection copyback or commits. This applies through helpers and value constructors. Large loops may use several launches; copyback waits for all to succeed. Native device writes and host callback side effects are not rolled back.
