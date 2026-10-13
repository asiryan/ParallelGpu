using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AdapterElementValidationRegressionTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void DiscardedSliceConstructorsRejectManagedElementRepresentationsBeforeWrites(int kind)
    {
        int[][] rows = [[3, 5], [7]];
        int[] output = [41];
        Action<int> body = kind switch
        {
            0 => i => { output[0] = 99; new ArraySegment<int[]>(rows); },
            1 => i => { output[0] = 99; new Memory<int[]>(rows); },
            2 => i => { output[0] = 99; new ReadOnlyMemory<int[]>(rows); },
            _ => i => { output[0] = 99; new Span<int[]>(rows); }
        };
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        Assert.Equal(new[] { 3, 5 }, rows[0]);
        Assert.Equal(new[] { 7 }, rows[1]);
        VerifyFollowingLoop(output);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DiscardedSliceExtensionsRejectManagedElementRepresentationsBeforeBridgeCompilation(bool memory)
    {
        int[][] rows = [[11, 13]];
        int[] output = [41];
        Action<int> body = memory
            ? i => { output[0] = 99; rows.AsMemory(); }
            : i => { output[0] = 99; rows.AsSpan(); };
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        VerifyFollowingLoop(output);
    }

    private void VerifyFollowingLoop(int[] output)
    {
        fixture.Gpu.For(0, 1, i => output[i] = i + 17);
        Assert.Equal(17, output[0]);
    }
}
