using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ScratchRegressionFixTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;
    private const int ChunkBoundary = 16 * 1024 * 1024 / 128;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ConstructorsWithTransitiveElementReferencesDeferInvalidAccess(bool nullArray)
    {
        long[] values = nullArray ? null! : [];
        int[] output = [41];
        Action<int> body = i => output[i] = new ConstructorRefValue(values, i).Number;
        if (nullArray)
            Assert.Throws<NullReferenceException>(() => Gpu.For(0, 1, body));
        else
            Assert.Throws<IndexOutOfRangeException>(() => Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);

        values = [7];
        Gpu.For(0, 1, body);
        Assert.Equal(17, output[0]);
        Assert.Equal(long.MaxValue, values[0]);
    }

    [Fact]
    public void ConstructorSafeReferencesIncludeLocallyConstructedMemoryViews()
    {
        long[] values = [];
        int[] output = [41];
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For(0, 1,
            i => output[i] = new SpanConstructorRefValue(values).Number));
        Assert.Equal(41, output[0]);
    }

    [Fact]
    public void LargeRangesWithReferencesReachDeferredValidationWithoutScratchOverflow()
    {
        // The old count * stride overflowed before any launch. An invalid access in the
        // first bounded launch verifies this range without executing hundreds of millions of bodies.
        WideReferenceValue[] values = [];
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For(0, 268435456, i =>
        {
            if (i == 0)
                GpuAtomic.Add(ref values[0].A, 1);
        }));
        int[] output = [41];
        Gpu.For(0, 1, i => output[i] = 17);
        Assert.Equal(17, output[0]);
    }

    [Fact]
    public void BoundedLaunchesPreserveIndicesSharedAtomicStorageAndTail()
    {
        const int start = -7;
        int length = ChunkBoundary + 3;
        WideReferenceValue[] values = [new() { A = 7 }];
        int[] output = new int[length];
        Gpu.For(start, start + length, i =>
        {
            GpuAtomic.Add(ref values[0].A, 1);
            output[i - start] = i;
        });
        Assert.Equal(7L + length, values[0].A);
        Assert.Equal(Enumerable.Range(start, length), output);
    }

    [Fact]
    public void AFailureInTheFinalLaunchDoesNotCommitEarlierLaunches()
    {
        int length = ChunkBoundary + 3;
        WideReferenceValue[] values = [new() { A = 7 }];
        int[] output = Enumerable.Repeat(41, length).ToArray();
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For(0, length, i =>
        {
            GpuAtomic.Add(ref values[0].A, 1);
            output[i] = 17;
            if (i == length - 1)
                GpuAtomic.Add(ref values[1].A, 1);
        }));
        Assert.Equal(7, values[0].A);
        Assert.All(output, value => Assert.Equal(41, value));
        Gpu.For(0, 1, i => output[i] = 23);
        Assert.Equal(23, output[0]);
    }

    private struct ConstructorRefValue
    {
        public int Number;
        public ConstructorRefValue(long[] values, int index)
        {
            Write(ref values[index]);
            Number = 17;
        }
        private static void Write(ref long value) => value = long.MaxValue;
    }

    private struct SpanConstructorRefValue
    {
        public int Number;
        public SpanConstructorRefValue(long[] values)
        {
            Write(values.AsSpan());
            Number = 17;
        }
        private static void Write(Span<long> values) => values[0] = long.MaxValue;
    }
}

public struct WideReferenceValue
{
    public long A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P;
}
