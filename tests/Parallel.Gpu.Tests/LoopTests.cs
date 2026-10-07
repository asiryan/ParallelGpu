using ILGPU;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class LoopTests(GpuFixture fixture)
{
    private GpuParallel Gpu => fixture.Gpu;

    [Fact]
    public void ForPreservesNeighborsAndUsesCurrentKernelArguments()
    {
        using var first = Gpu.Upload(Enumerable.Repeat(-1, 260).ToArray());
        using var second = Gpu.Upload(Enumerable.Repeat(-1, 260).ToArray());

        Gpu.For(1, 258, new WriteIndexKernel(first.View, 2));
        Gpu.For(1, 258, new WriteIndexKernel(second.View, 3));

        int[] firstResult = first.ToArray();
        int[] secondResult = second.ToArray();
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(i >= 1 && i < 258 ? i * 2 : -1, firstResult[i]);
            Assert.Equal(i >= 1 && i < 258 ? i * 3 : -1, secondResult[i]);
        }
    }

    [Fact]
    public void ForEachUpdatesArraysAndDeviceBuffers()
    {
        float[] values = [1, -2, 3];
        Gpu.ForEach(values, new ScaleKernel(2));
        Assert.Equal(new float[] { 2, -3, 8 }, values);

        using var buffer = Gpu.Upload(values);
        Gpu.ForEach(buffer, new ScaleKernel(3));
        Assert.Equal(new float[] { 6, -8, 26 }, buffer.ToArray());
    }

    [Fact]
    public void BuffersValidateCopiesOwnershipAndLifetime()
    {
        using var buffer = Gpu.Allocate<float>(3);
        Assert.Equal(new float[3], buffer.ToArray());
        Assert.Throws<ArgumentException>(() => buffer.CopyFrom(new float[2]));
        Assert.Throws<ArgumentException>(() => buffer.CopyTo(new float[4]));

        using var other = new GpuParallel(new() { Backend = GpuBackend.Cpu });
        Assert.Throws<ArgumentException>(() => other.ForEach(buffer, new ScaleKernel(2)));

        buffer.Dispose();
        Assert.Throws<ObjectDisposedException>(() => buffer.ToArray());
    }

    [Fact]
    public void DisposingOwnerReleasesItsBuffers()
    {
        using var owner = new GpuParallel(new() { Backend = GpuBackend.Cpu });
        var buffer = owner.Allocate<int>(3);
        owner.Dispose();

        Assert.Throws<ObjectDisposedException>(() => buffer.ToArray());
        Assert.Throws<ObjectDisposedException>(() => owner.Allocate<int>(1));
    }
}

public readonly struct WriteIndexKernel(ArrayView<int> data, int multiplier) : IGpuForKernel
{
    public void Execute(int index) => data[index] = index * multiplier;
}

public readonly struct ScaleKernel(float factor) : IGpuForEachKernel<float>
{
    public void Execute(int index, ref float value) => value = value * factor + index;
}
