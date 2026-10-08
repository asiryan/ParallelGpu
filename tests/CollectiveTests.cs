using ILGPU;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class CollectiveTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    private static void AppendThroughHelper(GpuAppendView<int> view, int value) => view.Add(value);

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(257)]
    [InlineData(65539)]
    public void ReductionHandlesEmptyTailsAndMultiplePasses(int length)
    {
        int[] input = Enumerable.Range(0, length).Select(i => i % 19 - 9).ToArray();
        int[] original = input.ToArray();
        int expected = input.Aggregate(0, (sum, value) => unchecked(sum + value));
        Assert.Equal(expected, Gpu.Reduce(input, new Int32Sum()));
        using var buffer = Gpu.Upload(input);
        Assert.Equal(expected, Gpu.Reduce(buffer, new Int32Sum()));
        Assert.Equal(original, input);
        Assert.Equal(original, buffer.ToArray());
    }

    [Fact]
    public void ReductionSupportsEveryAtomicNumericTypeAndCurrentOperationState()
    {
        Assert.Equal(6L, Gpu.Reduce(new long[] { 1, 2, 3 }, new Int64Sum()));
        Assert.Equal(6U, Gpu.Reduce(new uint[] { 1, 2, 3 }, new UInt32Sum()));
        Assert.Equal(6UL, Gpu.Reduce(new ulong[] { 1, 2, 3 }, new UInt64Sum()));
        Assert.Equal(6F, Gpu.Reduce(new float[] { 1, 2, 3 }, new Float32Sum()));
        Assert.Equal(6D, Gpu.Reduce(new double[] { 1, 2, 3 }, new Float64Sum()));
        int[] values = Enumerable.Range(0, 1031).Select(i => i % 7).ToArray();
        Assert.Equal(values.Sum() % 97, Gpu.Reduce(values, new ModuloReduction(97)));
        Assert.Equal(values.Sum() % 17, Gpu.Reduce(values, new ModuloReduction(17)));
    }

    [Fact]
    public void ReductionPreservesOrderForNonCommutativeAssociativeOperations()
    {
        var input = Enumerable.Range(0, 1031)
            .Select(i => new AffineValue((uint)(i % 3 + 1), (uint)i)).ToArray();
        var operation = new AffineReduction();
        var expected = input.Aggregate(operation.Identity, operation.Combine);
        var actual = Gpu.Reduce(input, operation);
        Assert.Equal(expected.Multiplier, actual.Multiplier);
        Assert.Equal(expected.Offset, actual.Offset);
    }

    [Fact]
    public void ReductionValidatesOwnershipAndLifetime()
    {
        using var buffer = Gpu.Upload(new int[] { 1, 2 });
        using var other = new Gpu(new() { Backend = GpuBackend.Cpu });
        Assert.Throws<ArgumentException>(() => other.Reduce(buffer, new Int32Sum()));
        Assert.Throws<ArgumentNullException>(() => Gpu.Reduce((int[])null!, new Int32Sum()));
        Assert.Throws<ArgumentNullException>(() => Gpu.Reduce((GpuBuffer<int>)null!, new Int32Sum()));
        buffer.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Gpu.Reduce(buffer, new Int32Sum()));
        other.Dispose();
        Assert.Throws<ObjectDisposedException>(() => other.Reduce(Array.Empty<int>(), new Int32Sum()));
    }

    [Fact]
    public void AtomicAddsWorkInOrdinaryLambdasForEveryNumericType()
    {
        int[] integers = [0];
        long[] longs = [0];
        uint[] unsigned = [0];
        ulong[] unsignedLongs = [0];
        float[] floats = [0];
        double[] doubles = [0];
        int[] previous = new int[1031];
        Gpu.For(0, previous.Length, i =>
        {
            previous[i] = GpuAtomic.Add(ref integers[0], 1);
            GpuAtomic.Add(ref longs[0], 1L);
            GpuAtomic.Add(ref unsigned[0], 1U);
            GpuAtomic.Add(ref unsignedLongs[0], 1UL);
            GpuAtomic.Add(ref floats[0], 1F);
            GpuAtomic.Add(ref doubles[0], 1D);
        });
        Assert.Equal(previous.Length, integers[0]);
        Assert.Equal((long)previous.Length, longs[0]);
        Assert.Equal((uint)previous.Length, unsigned[0]);
        Assert.Equal((ulong)previous.Length, unsignedLongs[0]);
        Assert.Equal((float)previous.Length, floats[0]);
        Assert.Equal((double)previous.Length, doubles[0]);
        Assert.Equal(Enumerable.Range(0, previous.Length), previous.OrderBy(value => value));
    }

    [Fact]
    public void AtomicMinMaxBitwiseAndCompareExchangeHaveDocumentedSemantics()
    {
        int[] minimum = [int.MaxValue];
        int[] maximum = [int.MinValue];
        int[] combined = [0];
        int[] masked = [-1];
        int[] parity = [0];
        Gpu.For(0, 32, i =>
        {
            GpuAtomic.Min(ref minimum[0], i - 13);
            GpuAtomic.Max(ref maximum[0], i - 13);
            GpuAtomic.Or(ref combined[0], 1 << (i % 16));
            GpuAtomic.And(ref masked[0], ~(1 << (i % 16)));
            GpuAtomic.Xor(ref parity[0], 1 << (i % 16));
        });
        Assert.Equal(-13, minimum[0]);
        Assert.Equal(18, maximum[0]);
        Assert.Equal(65535, combined[0]);
        Assert.Equal(unchecked((int)0xffff0000), masked[0]);
        Assert.Equal(0, parity[0]);
        int[] target = [10];
        int[] result = new int[3];
        Gpu.For(0, 1, i =>
        {
            result[0] = GpuAtomic.CompareExchange(ref target[0], 20, 10);
            result[1] = GpuAtomic.CompareExchange(ref target[0], 30, 10);
            result[2] = GpuAtomic.Exchange(ref target[0], 40);
        });
        Assert.Equal(new int[] { 10, 20, 20 }, result);
        Assert.Equal(40, target[0]);
    }

    [Fact]
    public void CollectAppendsConstructedValuesAndSupportsEmptyOutput()
    {
        int[] expected = Enumerable.Range(-9, 21).Where(i => i % 2 == 0).Select(i => i * i).OrderBy(i => i).ToArray();
        int[] actual = Gpu.Collect<int, SelectEvenKernel>(-9, 12, expected.Length, new());
        Assert.Equal(expected, actual.OrderBy(i => i));
        Assert.Empty(Gpu.Collect<int, SelectEvenKernel>(0, 0, 0, new()));
        Assert.Empty(Gpu.Collect<int, SelectEvenKernel>(1, 2, 0, new()));
        var pairs = Gpu.Collect<AffineValue, ConstructValuesKernel>(0, 1031, 1031, new());
        Assert.Equal(Enumerable.Range(0, 1031).Select(i => (uint)i), pairs.Select(value => value.Offset).OrderBy(value => value));
        Assert.All(pairs, value => Assert.Equal(2U, value.Multiplier));
    }

    [Fact]
    public void BoundedOutputOverflowsSafelySaturatesCountAndCanReset()
    {
        using var output = Gpu.CreateAppendBuffer<int>(17);
        Gpu.For(0, 1031, new AppendIndexesKernel(output.View));
        Assert.True(output.Overflowed);
        Assert.Equal(17, output.Count);
        Assert.Throws<InvalidOperationException>(() => output.ToArray());
        output.Reset();
        Assert.False(output.Overflowed);
        Assert.Equal(0, output.Count);
        Assert.Empty(output.ToArray());
        Gpu.For(0, 17, new AppendIndexesKernel(output.View));
        Assert.False(output.Overflowed);
        Assert.Equal(Enumerable.Range(0, 17), output.ToArray().OrderBy(value => value));
        Assert.Throws<InvalidOperationException>(() => Gpu.Collect<int, SelectEvenKernel>(0, 35, 17, new()));
        Assert.Throws<InvalidOperationException>(() => Gpu.Collect<int, SelectEvenKernel>(0, 1, 0, new()));
    }

    [Fact]
    public void OrdinaryLambdaCanCaptureBoundedAppendView()
    {
        using var output = Gpu.CreateAppendBuffer<int>(1031);
        var view = output.View;
        Gpu.For(0, 1031, i => AppendThroughHelper(view, i * 3));
        Assert.Equal(Enumerable.Range(0, 1031).Select(i => i * 3), output.ToArray().OrderBy(value => value));
    }

    [Fact]
    public void NativeAppendCaptureRejectsMappedCharacterPayloadBeforeLaunchAndExecutorRecovers()
    {
        int[] markers = [0];
        Assert.Throws<NotSupportedException>(() => Gpu.CreateAppendBuffer<char>(1));
        GpuAppendView<char> characterView = default;
        Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i =>
        {
            markers[i] = 99;
            characterView.Add('\u9fa5');
        }));
        Assert.Equal(0, markers[0]);
        using var supported = Gpu.CreateAppendBuffer<ushort>(1);
        var unicodeView = supported.View;
        Gpu.For(0, 1, i =>
        {
            markers[i] = 7;
            unicodeView.Add(0x9fa5);
        });
        Assert.Equal(7, markers[0]);
        Assert.Equal(new ushort[] { 0x9fa5 }, supported.ToArray());
    }

    [Fact]
    public void NativeAppendCapturesValidateLifetimeOwnerAndDefaultBeforeEffects()
    {
        var markers = new int[1];
        var output = Gpu.CreateAppendBuffer<int>(1);
        var stale = output.View;
        Action<int> body = i =>
        {
            markers[i] = 99;
            stale.Add(i);
        };
        Gpu.For(0, 1, body);
        Assert.Equal(99, markers[0]);
        output.Dispose();
        markers[0] = 0;
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, 1, body));
        Assert.Equal(0, markers[0]);
        using var foreignOwner = new Gpu(new() { Backend = GpuBackend.Cpu });
        using var foreignOutput = foreignOwner.CreateAppendBuffer<int>(1);
        var foreign = foreignOutput.View;
        Assert.Throws<ArgumentException>(() => Gpu.For(0, 1, i =>
        {
            markers[i] = 99;
            foreign.Add(i);
        }));
        Assert.Equal(0, markers[0]);
        GpuAppendView<int> empty = default;
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, 1, i =>
        {
            markers[i] = 99;
            empty.Add(i);
        }));
        Assert.Equal(0, markers[0]);
        var ownedOutput = foreignOwner.CreateAppendBuffer<int>(1);
        var ownedView = ownedOutput.View;
        foreignOwner.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Gpu.For(0, 1, i => ownedView.Add(i)));
        ownedOutput.Dispose();
    }

    [Fact]
    public void AppendReservationsAreUniqueAndOutputLifetimeIsOwned()
    {
        using var output = Gpu.CreateAppendBuffer<int>(1031);
        using var indices = Gpu.Allocate<int>(1031);
        Gpu.For(0, 1031, new AppendWithIndexKernel(output.View, indices.View));
        var values = output.ToArray();
        var reservations = indices.ToArray();
        Assert.Equal(Enumerable.Range(0, 1031), reservations.OrderBy(value => value));
        for (int i = 0; i < reservations.Length; i++)
            Assert.Equal(i, values[reservations[i]]);
        output.Dispose();
        Assert.Throws<ObjectDisposedException>(() => output.ToArray());
        Assert.Throws<ObjectDisposedException>(() => output.Reset());
        using var owner = new Gpu(new() { Backend = GpuBackend.Cpu });
        using var ownedOutput = owner.CreateAppendBuffer<int>(1);
        owner.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ownedOutput.View);
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.CreateAppendBuffer<int>(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.Collect<int, SelectEvenKernel>(0, 1, -1, new()));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gpu.Collect<int, SelectEvenKernel>(int.MinValue, int.MaxValue, 1, new()));
    }
}

public readonly struct ModuloReduction(int modulus) : IGpuReduction<int>
{
    public int Identity => 0;
    public int Combine(int left, int right) => (left + right) % modulus;
}

public readonly struct AffineValue(uint multiplier, uint offset)
{
    public readonly uint Multiplier = multiplier;
    public readonly uint Offset = offset;
}

public readonly struct AffineReduction : IGpuReduction<AffineValue>
{
    public AffineValue Identity => new(1, 0);
    public AffineValue Combine(AffineValue left, AffineValue right) =>
        new(unchecked(right.Multiplier * left.Multiplier), unchecked(right.Multiplier * left.Offset + right.Offset));
}

public readonly struct SelectEvenKernel : IGpuCollectKernel<int>
{
    public void Execute(int index, GpuAppendView<int> output)
    {
        if (index % 2 == 0)
            output.Add(index * index);
    }
}

public readonly struct ConstructValuesKernel : IGpuCollectKernel<AffineValue>
{
    public void Execute(int index, GpuAppendView<AffineValue> output) => output.Add(new AffineValue(2, (uint)index));
}

public readonly struct AppendIndexesKernel(GpuAppendView<int> output) : IGpuForKernel
{
    public void Execute(int index) => output.Add(index);
}

public readonly struct AppendWithIndexKernel(GpuAppendView<int> output, ArrayView<int> indices) : IGpuForKernel
{
    public void Execute(int index)
    {
        if (output.TryAdd(index, out int reserved))
            indices[index] = reserved;
    }
}
