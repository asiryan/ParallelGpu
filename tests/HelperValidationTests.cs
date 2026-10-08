using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class HelperValidationTests(GpuFixture fixture)
{
    private static readonly int[] StaticInput = [3];

    [Fact]
    public void ManagedHelperEffectsAreRejectedWithoutPoisoningLaterLoops()
    {
        int[] output = [-1];
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i => output[i] = Boxed(i)));
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i => output[i] = Allocated(i)));
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i => output[i] = StaticRead(i)));
        Assert.Equal(-1, output[0]);
        fixture.Gpu.For(0, 1, i => output[i] = StaticInput[i] + 4);
        Assert.Equal(7, output[0]);
    }

    private static int Boxed(int value) { object boxed = value; return (int)boxed; }
    private static int Allocated(int value) => new int[value + 1].Length;
    private static int StaticRead(int index) => StaticInput[index];
}
