# ParallelGpu

Synchronous C# loops on [CUDA](https://developer.nvidia.com/cuda) or [OpenCL](https://www.khronos.org/opencl/), powered by [ILGPU](https://ilgpu.net). Write a lambda over captured collections; the library compiles it, transfers the data, and copies results back before returning.

## Installation

```shell
dotnet add package ParallelGpu
```

## Quick start

```csharp
using ParallelGpu;

using var gpu = new Gpu();
int[] input = { 1, 2, 3 };
int[] output = new int[input.Length];

gpu.For(0, input.Length, i => output[i] = input[i] * 2);
// output: 2, 4, 6
```

Reuse a `Gpu` instance to retain the accelerator and compiled kernels. `GpuParallel.For(...)` provides a static alternative.

## Backend selection

| Backend | Requirements and behavior |
| --- | --- |
| `Auto` (default) | Tries CUDA, then OpenCL. Set `AllowCpuFallback = true` in `GpuOptions` to allow CPU fallback. |
| `Cuda` | Requires an NVIDIA GPU and a compatible CUDA driver. |
| `OpenCL` | Requires a GPU runtime with OpenCL C 2.0 or newer and generic address spaces; an OpenCL 3.0 version label alone is insufficient. |
| `Cpu` | Uses the ILGPU CPU accelerator for debugging and correctness checks. |

Select a backend with `new Gpu(new GpuOptions { Backend = GpuBackend.Cuda })`. Explicit `Cuda` and `OpenCL` selections never fall back to CPU. `gpu.Backend` and `gpu.DeviceName` identify the selected device.

The test suite passes on CPU and CUDA. OpenCL has not been verified in the current test environment.

## Capabilities

- Arrays of any rank, jagged arrays, fixed-size list operations, and supported memory views, preserving shared storage.
- Numeric types, `bool`, `char`, enums, supported value structures, and nullable values.
- Persistent device buffers, explicit value kernels, reductions, atomics, and bounded append buffers.
- Custom collection and object-graph adapters, plus supported .NET 8 `Math`/`MathF` operations.

## Execution requirements and limitations

The automatic lambda API requires runtime code generation (`Reflection.Emit`) and access to method IL. Use an ordinary `void Method(int)` delegate; Native AOT and `DynamicMethod` delegates are unsupported.

Loop iterations are unordered. Use independent writes, atomics, or reductions to avoid races. Captured scalars are read-only snapshots. Collection elements may be writable, but sizes and references between arrays cannot change during a loop.

Loop bodies and their helpers must be device-compatible. Arbitrary .NET methods, managed allocation, boxing, reflection, exception handlers, and checked integer arithmetic are unsupported. Value structures must satisfy the [supported type rules](https://github.com/asiryan/ParallelGpu/blob/main/docs/guide.md); device buffers and explicit kernels have stricter ILGPU type requirements.

Automatic loops transfer captured collections on every call. For repeated operations, keep data in device buffers. Transfers and compilation can outweigh GPU gains for small workloads.

`GpuView<T>` borrows device memory: keep its owner alive and use valid indices. Detected loop errors prevent automatic collection copyback, but device buffer writes and host callback effects are not rolled back.

Calls on one `Gpu` instance are serialized. Avoid concurrent host changes to transferred collections, and dispose the executor and its buffers when finished. Floating-point results may differ between devices; transcendental functions and reductions require suitable numerical tolerances.

## Documentation and examples

See the [API guide](https://github.com/asiryan/ParallelGpu/blob/main/docs/guide.md), [supported math operations](https://github.com/asiryan/ParallelGpu/blob/main/docs/math.md), and [sample applications](https://github.com/asiryan/ParallelGpu/tree/main/samples).

## License

Licensed under MIT.
