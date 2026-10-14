<p align="center"><img width="45%" src="docs/logo_big.png" /></p>

Synchronous C# loops on [CUDA](https://developer.nvidia.com/cuda) or [OpenCL](https://www.khronos.org/opencl/), powered by [ILGPU](https://ilgpu.net). Write a lambda over captured collections; the library compiles it, transfers the data, and copies results back before returning.

# Version

Build **ParallelGpu** from source or install the NuGet package in your project.

| Assembly | Specification | OS | Download | Package |
|-------------|:-------------:|:-------------:|:--------------:|:--------------:|
| [ParallelGpu](sources) | .NET Standard 2.1 | Cross-platform | [Release](https://github.com/asiryan/ParallelGpu/releases/) | [NuGet](https://www.nuget.org/packages/ParallelGpu/) | 

# Installation

```shell
dotnet add package ParallelGpu
```

The library targets .NET Standard 2.1. Install from [NuGet](https://www.nuget.org/packages/ParallelGpu/) or build from source; ILGPU dependencies are restored automatically.

```csharp
using ParallelGpu;

using var gpu = new Gpu();
int[] input = { 1, 2, 3 };
int[] output = new int[input.Length];

gpu.For(0, input.Length, i => output[i] = input[i] * 2);
// output: 2, 4, 6
```

Reuse a `Gpu` instance to retain the device and compiled kernels. `GpuParallel.For(...)` provides a static API; use `Gpu` for repeated calls with custom options.

## Choose a backend

| `GpuOptions.Backend` | Behavior |
| --- | --- |
| `Auto` (default) | Tries CUDA, then OpenCL. CPU fallback requires `AllowCpuFallback = true`. |
| `Cuda` | Requires an NVIDIA GPU and a compatible CUDA driver. |
| `OpenCL` | Requires OpenCL C 2.0 or newer with generic address-space support. An OpenCL 3.0 label alone is insufficient. |
| `Cpu` | Uses the ILGPU CPU accelerator for debugging and correctness checks. |

```csharp
using var gpu = new Gpu(new GpuOptions
{
    Backend = GpuBackend.Auto,
    AllowCpuFallback = true
});
```

Explicit `Cuda` and `OpenCL` selections never fall back to CPU. `DeviceIndex` selects a device; `gpu.Backend`, `gpu.DeviceName`, and `gpu.IsGpu` report the selection. CPU and CUDA are tested. OpenCL remains unverified because the local driver lacks the required generic address-space support.

## What can run on the device?

| Feature | API and details |
| --- | --- |
| Automatic collection transfers | Arrays of any rank, jagged arrays, lists, and memory slices; shared storage is preserved. See [supported values and collections](docs/guide.md#values-and-collections). |
| Data kept on the device | `Upload<T>`, `Allocate<T>`, and `GpuBuffer<T>.DeviceView`; see the example below. |
| Reductions and synchronization | `Reduce`, built-in sums, and `GpuAtomic`; see [reductions and atomics](docs/guide.md#reductions-and-atomics). |
| Variable output | Bounded append buffers and `Collect`; see [bounded output](docs/guide.md#bounded-output). |
| Custom data models | [Collection adapters](docs/guide.md#collection-adapters) and [object graphs](docs/guide.md#object-graphs). |
| Numeric methods | [Supported Math and MathF overloads](docs/math.md). |
| Explicit kernels | Public value types implementing `IGpuForKernel` or `IGpuForEachKernel<T>`; see [native buffers and kernels](docs/guide.md#native-buffers-and-kernels). |

Loop bodies and helpers must use device-compatible operations. Managed allocation, boxing, reflection, exception handlers, checked integer arithmetic, and arbitrary .NET calls are unsupported. Automatic lambda compilation requires `Reflection.Emit` and method IL, so Native AOT is unsupported.

Iterations are unordered: use independent writes, atomics, or reductions to avoid races. Scalar captures are read-only snapshots; collection sizes cannot change during a loop. See the [execution rules](docs/guide.md#execution-rules) for error handling and other constraints.

## Keep data on the device

Automatic loops transfer captured collections on every call. For repeated work, upload once and download when needed. This example reuses `gpu` from the quick start:

```csharp
using var buffer = gpu.Upload(new float[] { 1, 2, 3 });
var values = buffer.DeviceView;

gpu.For(0, values.Length, i => values[i] *= 2);
gpu.For(0, values.Length, i => values[i] *= 3);
float[] scaled = buffer.ToArray();
// scaled: 6, 12, 18
```

Native buffers use unmanaged ILGPU-compatible types, without automatic value mappings. Views borrow their owner's memory; keep the owner alive, use its executor, and keep indices in range. See [buffer types, transfers, and lifetime](docs/guide.md#native-buffers-and-kernels).

Transfers and compilation can outweigh GPU execution for small workloads. Measure complete calls after warm-up in Release mode.

# Examples

The [samples](samples) cover [matrix multiplication](samples/FloatMatrixMultiplication), [FFT](samples/FastFourierTransform), [hash search](samples/PasswordHashSearch), [Monte Carlo](samples/MonteCarloPi), [factorization](samples/SemiPrimeFactorization), [digit recognition](samples/DigitRecognizer), [Mandelbrot](samples/Mandelbrot), [ray tracing](samples/RayTracer), and [image denoising](samples/ImageDenoising).

Run a sample with the .NET 8 SDK:

```shell
dotnet run --project samples/FloatMatrixMultiplication -c Release
```

Digit recognition requires `.artifacts/digit-recognizer.zip`; dataset and training settings are in its [Program.cs](samples/DigitRecognizer/Program.cs). Image denoising includes an [input image](samples/ImageDenoising/Assets/README.md).

# Build and test

```powershell
dotnet build ParallelGpu.sln -c Release -warnaserror
$env:PARALLEL_GPU_TEST_BACKEND = 'Cpu'
dotnet test tests/ParallelGpu.Tests.csproj -c Release --no-build
$env:PARALLEL_GPU_TEST_BACKEND = 'Cuda'
dotnet test tests/ParallelGpu.Tests.csproj -c Release --no-build
```

Tests default to `Cpu`; the environment variable also accepts `Cuda`, `OpenCL`, and `Auto` (`Auto` requires a GPU). `build.bat` builds just the library and NuGet package. Release packages in `sources/bin/Release/` include the license, package README, logo, and XML API documentation.

# License

[MIT](LICENSE)
