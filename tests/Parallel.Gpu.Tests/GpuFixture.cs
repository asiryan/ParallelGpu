using ParallelGpu;
using Xunit;

namespace ParallelGpu.Tests;

public sealed class GpuFixture : IDisposable
{
    public Gpu Gpu { get; }

    public GpuFixture()
    {
        string requested = Environment.GetEnvironmentVariable("PARALLEL_GPU_TEST_BACKEND") ?? "Cpu";
        if (!Enum.TryParse<GpuBackend>(requested, true, out var backend) || !Enum.IsDefined(backend))
            throw new ArgumentException($"Unknown test backend: {requested}.");
        Gpu = new(new() { Backend = backend });
        Console.WriteLine($"Test backend: {Gpu.Backend}; device: {Gpu.DeviceName}; GPU: {Gpu.IsGpu}");
        if (backend != GpuBackend.Auto)
            Assert.Equal(backend, Gpu.Backend);
        if (backend is GpuBackend.Cuda or GpuBackend.OpenCL or GpuBackend.Auto)
            Assert.True(Gpu.IsGpu);
    }

    public void Dispose() => Gpu.Dispose();
}

[CollectionDefinition("GPU")]
public sealed class GpuCollection : ICollectionFixture<GpuFixture>;
