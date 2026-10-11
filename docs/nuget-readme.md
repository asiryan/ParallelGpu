# ParallelGpu

Synchronous C# loops on [CUDA](https://developer.nvidia.com/cuda) or [OpenCL](https://www.khronos.org/opencl/), powered by [ILGPU](https://ilgpu.net). Write a lambda over captured collections; the library compiles it for the selected device, transfers the data, and copies results back before the call returns.

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

Reuse a `Gpu` instance to retain the accelerator and compiled kernels. `GpuParallel.For(...)` also provides a static API.

## Backend selection

The default backend is `Auto`. It tries CUDA first, then OpenCL. CPU fallback is disabled by default; enable it explicitly when an application should also run without a compatible GPU:

```csharp
using var gpu = new Gpu(new GpuOptions
{
    Backend = GpuBackend.Auto,
    AllowCpuFallback = true
});
```

| Backend | Requirements and behavior |
| --- | --- |
| `Auto` | Tries available GPUs in CUDA-first order; CPU fallback requires `AllowCpuFallback = true`. |
| `Cuda` | Requires an NVIDIA GPU and a compatible CUDA driver. |
| `OpenCL` | Requires a compatible GPU and OpenCL runtime with OpenCL C 2.0 or newer and generic address-space support. An OpenCL 3.0 version label alone is insufficient. |
| `Cpu` | Uses the ILGPU CPU accelerator for debugging and correctness checks. |

Explicit `Cuda` and `OpenCL` selections never fall back to CPU. Use `GpuOptions.DeviceIndex` to select a device, and inspect `gpu.Backend`, `gpu.DeviceName`, and `gpu.IsGpu` after initialization.

## Capabilities

- Automatic transfers for arrays of any rank, jagged arrays, fixed-size lists, and supported memory views, with shared storage preserved.
- Automatic value mappings for numeric types, `bool`, `char`, enums, supported structures, and nullable values.
- Device buffers via `Upload<T>` and `Allocate<T>` to retain data between kernels.
- Explicit value kernels through `IGpuForKernel` and `IGpuForEachKernel<T>`.
- Reductions, atomic operations, and bounded append buffers for variable output.
- Host collection adapters and explicit object-graph adapters for custom data models.

## Execution requirements and limitations

The automatic lambda API requires runtime code generation (`Reflection.Emit`) and access to method IL. It is not supported in [Native AOT](https://learn.microsoft.com/en-us/dotnet/core/deploying/native-aot/) applications.

Loop iterations are unordered. Use independent writes, atomics, or reductions to avoid races. Scalars captured by a lambda are read-only; collection elements may be writable, but collection sizes and array topology cannot change during execution.

Loop bodies and their helpers must be device-compatible. Arbitrary .NET methods, managed allocation, boxing, reflection, exception handlers, and checked integer arithmetic are unsupported. Automatic structure mappings cannot contain managed references, native pointers, or explicit layout. Inline arrays and fixed buffers are unsupported, including in nested structures. Direct device buffers and explicit value kernels require public unmanaged ILGPU-compatible types.

Automatic loops transfer captured collections on every call. For repeated operations, keep data in device buffers and download results when needed. Transfers and kernel compilation can outweigh the benefit of GPU execution for small or inexpensive workloads.

Calls on one `Gpu` instance are serialized. Avoid modifying transferred host collections concurrently, and dispose the executor and its buffers when finished. Floating-point reduction results may differ from sequential CPU results and between devices.

## Documentation and examples

See the [ParallelGpu repository](https://github.com/asiryan/ParallelGpu) for the full API guide, supported math operations, collection and object-graph examples, and sample applications.

## License

Licensed under MIT.
