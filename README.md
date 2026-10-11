<p align="center"><img width="15%" src="docs/logo.png" /></p>  

# ParallelGpu

Synchronous C# loops on CUDA or OpenCL, powered by ILGPU. Write an ordinary lambda; the library compiles it, transfers captured collections, and copies results back before returning.

The library targets .NET Standard 2.1. Build the solution and samples with the .NET 8 SDK. Reference `sources/ParallelGpu.csproj` from your application.

## Quick start

```csharp
using System;
using ParallelGpu;

using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
int[] input = { 1, 2, 3 };
int[] output = new int[input.Length];
gpu.For(0, input.Length, i => output[i] = input[i] * 2);
// output: 2, 4, 6
```

Reuse `Gpu` to reuse the device and compiled kernels. `GpuParallel.For(...)` offers the same lambda API: calls without options share an executor; nonempty calls with options create one per call. The examples below reuse `gpu` from above.

| Backend | Selection |
| --- | --- |
| `Auto` | CUDA first, then OpenCL; CPU fallback requires `AllowCpuFallback = true` |
| `Cuda` | NVIDIA GPU with an installed CUDA driver |
| `OpenCL` | Compatible GPU and OpenCL runtime |
| `Cpu` | ILGPU CPU accelerator for debugging and correctness checks |

Explicit GPU selections never fall back to CPU. Inspect `gpu.Backend`, `gpu.DeviceName`, and `gpu.IsGpu` to identify the selected device. OpenCL requires OpenCL C 2.0 or newer and generic address-space support; an OpenCL 3.0 version label alone is insufficient.

## Supported values and collections

Automatic lambda loops support:

| Values | Contract |
| --- | --- |
| `byte`, `sbyte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong` | Standard integer types |
| `float`, `double`, `bool`, `char` | Includes full UTF-16 character values |
| `enum` | Any standard integer underlying type |
| `struct` | Supported fields recursively, including private, readonly, nested, and generic structures |
| `Nullable<T>` | Any supported payload type |

Structures cannot contain managed references, native pointers, or explicit layout. `Complex` and value tuples are supported when their executed methods are device-compatible. Supporting a type does not imply support for every framework method on it.

| Collections and views | Contract |
| --- | --- |
| `T[]`, `T[,]`, `T[,,]`, higher ranks | Any CLR array rank; dimensions and nonzero lower bounds are retained |
| `T[][]`, deeper nesting, mixed rectangular/jagged arrays | Null branches, empty dimensions, and shared references are retained |
| `List<T>` (also via `IList<T>` or `IReadOnlyList<T>`) | List element access; fixed size during execution |
| `ArraySegment<T>`, `Memory<T>`, `ReadOnlyMemory<T>` | Capturable slices; overlapping views share storage |
| `Span<T>`, `ReadOnlySpan<T>` | Local values and helper parameters; integer indexing and slicing |

`For<T>` requires all automatically transferred collections to have ultimate element type `T`. Use `For(...)` for mixed element types. Device-compatible helpers may receive and return supported collections and references to elements.

```csharp
int[,,] volume = new int[2, 3, 4];
gpu.For<int>(0, volume.GetLength(0), x =>
{
    for (int y = 0; y < volume.GetLength(1); y++)
        for (int z = 0; z < volume.GetLength(2); z++)
            volume[x, y, z] = x * 100 + y * 10 + z;
});

int?[] values = { null, 2, 3 };
int? offset = 4;
gpu.For<int?>(0, values.Length, i => values[i] += offset);
// values: null, 6, 7
```

Memory views expose integer `Slice`, `Span`, `Length`, and `IsEmpty` where available; segments expose `Count`. Array `AsSpan`/`AsMemory` and supported constructors/conversions work in helpers. `Index`/`Range` slicing and arbitrary collection methods are unsupported. String-backed `ReadOnlyMemory<char>` is readable; custom memory managers use host snapshots and commits.

```csharp
int[] storage = { 1, 2, 3, 4 };
Memory<int> slice = storage.AsMemory(1, 2);
ReadOnlyMemory<int> readOnly = storage.AsMemory(1, 2);
int[] observed = new int[2];
gpu.For(0, slice.Length, i =>
{
    slice.Span[i] += 10;
    observed[i] = readOnly.Span[i];
});
// storage: 1, 12, 13, 4; observed: 12, 13
```

## Keep data on the device

Automatic loops transfer captured collections on each call. Slices transfer their full backing owner to preserve aliases. For repeated work, upload once, run several kernels, and download once:

```csharp
using var buffer = gpu.Upload(new float[] { 1, 2, 3 });
gpu.ForEach(buffer, new ScaleKernel(2));
gpu.ForEach(buffer, new ScaleKernel(3));
float[] scaled = buffer.ToArray();
// scaled: 6, 12, 18

public readonly struct ScaleKernel(float factor) : IGpuForEachKernel<float>
{
    public void Execute(int index, ref float value) => value *= factor;
}
```

`Allocate<T>(length)` creates zero-filled storage. Direct buffers and explicit value kernels require public unmanaged ILGPU-compatible types; they do not apply automatic mappings for private structures, `bool`, `char`, or nullable values.

GPU execution can be slower for cheap operations when transfers dominate. Measure complete calls after warm-up, and compare equivalent algorithms in Release mode.

## Reductions and atomics

```csharp
int total = gpu.Reduce(new int[] { 1, 2, 3 }, new Int32Sum());
// total: 6

int[] counter = { 0 };
gpu.For(0, 1000, i => GpuAtomic.Add(ref counter[0], 1));
// counter[0]: 1000
```

Built-in sums cover `int`, `long`, `uint`, `ulong`, `float`, and `double`; `Reduce` also accepts an uploaded buffer. Custom `IGpuReduction<T>` implementations need a pure associative `Combine` and a two-sided `Identity`. Input order is preserved; empty input returns `Identity`. Integer sums use unchecked overflow. Floating-point results may differ from sequential CPU sums and between devices.

`GpuAtomic` provides `Add`, `Min`, `Max`, `Exchange`, and `CompareExchange` for those six types, plus integer `And`, `Or`, and `Xor`. Operations return the previous value. Compare-exchange uses the same argument order as `Interlocked.CompareExchange`.

## Bounded output

Use an append buffer when iterations produce a variable number of results:

```csharp
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

Capacity is fixed. `TryAdd(value, out index)` reports success and the reserved slot; on failure, `index` is `-1` and overflow is recorded. `ToArray()` rejects overflowed output. Call `Reset()` before reuse, and read produced values only after the producing kernel completes. `Collect<T,TKernel>` is the equivalent one-call API for `IGpuCollectKernel<T>`.

## Collection adapters and object graphs

- **Custom indexed collections:** register `GpuCollectionAdapters.Register<TCollection,T>` before compilation and supply a `GpuCollectionSnapshot<T>`. A snapshot describes the full buffer, slice, read-only status, shared identity, and optional host commit. Shared identities preserve aliases and reuse one commit per invocation. See [adapter and memory-manager examples](tests/CollectionAdapterTests.cs).
- **Classes, shared objects, and cycles:** use `UploadObjectGraph<TObject,TRecord>` with `IGpuObjectGraphAdapter<TObject,TRecord>`. The host class becomes a public unmanaged record plus explicit links. Reserve record/reference capacity for device-side creation. `Download()` returns records; `CopyToHost()` updates existing objects and reconstructs new ones. See [object-graph examples](tests/ObjectGraphTests.cs).

Adapters run on the host; callback failures can leave partial host changes. Graph roots and links may change, but capacities and each record's reference count are fixed. Newly created records become readable by other iterations in a later kernel. Graph handles belong to their graph. Graph errors make downloads fail; dispose the failed graph and upload a new one. Download append/graph results explicitly to validate their error flags.

## Framework math

Only the following `Math`/`MathF` signatures are supported. Parameters and return values use the stated type.

| Methods | Types |
| --- | --- |
| `Math.Abs` | `sbyte`, `short`, `int`, `long`, `float`, `double` |
| `Math.Min`, `Math.Max`, `Math.Clamp` | The ten numeric types in the values table |
| `Math.Sqrt`, `Math.Floor`, `Math.Ceiling` | `double` |
| `Math.CopySign` | `double` |
| `MathF.Abs`, `MathF.Sqrt`, `MathF.Floor`, `MathF.Ceiling` | `float` |
| `MathF.CopySign`, `MathF.Min`, `MathF.Max` | `float` |

Floating-point `Min`/`Max` preserve .NET NaN and signed-zero behavior; `Clamp` uses ordered comparisons. Other intrinsic rounding can differ from CPU execution. Unsupported calls, including `Sin`, `Cos`, `Pow`, `Log`, and `Round`, are rejected before device compilation, including calls through helpers.

## Execution rules and limitations

The automatic lambda API requires runtime code generation (`Reflection.Emit`) and access to method IL. It is not supported in [Native AOT](https://learn.microsoft.com/en-us/dotnet/core/deploying/native-aot/) applications.

- Loops cover `[fromInclusive, toExclusive)` with at most `Int32.MaxValue` iterations. Empty and reversed ranges execute no body. Iterations are unordered; independent writes, atomics, or reductions must prevent races.
- Scalar captures are refreshed per call and are read-only in the body. Collection elements are writable; read-only views reject writes. Array/list topology cannot change during an automatic loop.
- Bounds, null access, absent nullable `Value`, read-only writes, integer division errors, signed `Abs` overflow, and inverted `Clamp` bounds become host exceptions before automatic collection copyback. Owned device buffers and callback side effects are not rolled back. Explicit value kernels follow ILGPU's execution contracts.
- Calls on one executor are serialized. Do not change transferred host collections concurrently. Dispose buffers, append buffers, graphs, and the executor; borrowed views must stay within their owner's lifetime and use the same executor. Native append/graph payloads must need no automatic conversion; use `ushort` for characters and `int` for Boolean fields.
- Unsupported operations include direct managed/reference-bearing values, boxing, reflection, managed allocation, throwing bodies/exception handlers, checked integer arithmetic/conversions, dynamic managed dispatch, delegates, native pointers, and `ref struct` storage. Helpers remain subject to ILGPU and backend restrictions.

## Build and test

The library creates a NuGet package on build. Release packages are written to `sources/bin/Release/` and include the MIT license, package README, logo, and XML API documentation. Run `build.bat` to build the library and package, or use the solution commands below to build and test all projects.

```powershell
dotnet build ParallelGpu.sln -c Release -warnaserror
dotnet test tests/ParallelGpu.Tests.csproj -c Release
$env:PARALLEL_GPU_TEST_BACKEND = 'Cuda'
dotnet test tests/ParallelGpu.Tests.csproj -c Release
dotnet run --project samples/FloatMatrixMultiplication -c Release
dotnet run --project samples/ComplexVectorMultiplication -c Release
```

Tests default to the CPU backend. Set `PARALLEL_GPU_TEST_BACKEND` to `Cpu`, `Cuda`, `OpenCL`, or `Auto` to select a path; `Auto` requires a GPU. CPU and CUDA passed the 182-test audit. OpenCL remains unverified: the local RTX 4070 Ti driver lacks generic address-space support and cannot initialize the ILGPU OpenCL accelerator.

## License

**MIT**
