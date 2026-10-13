using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class OptimizationSafetyTransferTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CaptureCallbacksCannotExpireAppendOrGraphViewsBeforeTheOuterLaunch(bool graphView)
    {
        Indexed.Register();
        var collection = new Indexed([7]);
        int[] output = [41];
        if (graphView)
        {
            using var graph = fixture.Gpu.UploadObjectGraph<GraphTestNode, GraphTestRecord>(
                [new GraphTestNode(1)], new GraphTestAdapter());
            var view = graph.View;
            collection.CaptureAction = graph.Dispose;
            Assert.Throws<ObjectDisposedException>(() => fixture.Gpu.For(0, 1,
                i => output[i] = collection[i] + view.RootCount));
            Assert.Throws<ObjectDisposedException>(() => graph.Download());
        }
        else
        {
            using var append = fixture.Gpu.CreateAppendBuffer<int>(1);
            var view = append.View;
            collection.CaptureAction = append.Dispose;
            Assert.Throws<ObjectDisposedException>(() => fixture.Gpu.For(0, 1,
                i => output[i] = collection[i] + view.Capacity));
            Assert.Throws<ObjectDisposedException>(() => append.ToArray());
        }
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(new[] { 7 }, collection.Buffer);
        Assert.Equal(1, collection.CaptureCount);
        Assert.Equal(0, collection.CommitCount);
        fixture.Gpu.For(0, 1, i => output[i] = 17);
        Assert.Equal(new[] { 17 }, output);
    }

    [Fact]
    public void AbortedCaptureWithNestedDeviceFailureDoesNotPoisonCachedRecovery()
    {
        Indexed.Register();
        int[] empty = [], nestedOutput = [43], output = [41];
        var collection = new Indexed([7])
        {
            CaptureAction = () => fixture.Gpu.For(0, 1, i => nestedOutput[i] = empty[i])
        };
        Action<int> body = i =>
        {
            collection[i]++;
            output[i] = collection[i];
        };
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(new[] { 43 }, nestedOutput);
        Assert.Equal(new[] { 7 }, collection.Buffer);
        Assert.Equal(0, collection.CommitCount);

        collection.CaptureAction = null;
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(new[] { 8 }, output);
        Assert.Equal(new[] { 8 }, collection.Buffer);
        Assert.Equal(2, collection.CaptureCount);
        Assert.Equal(1, collection.CommitCount);
    }

    [Fact]
    public void LaterBatchCommitsStillRunAfterAnEarlierCommitCatchesANestedFailure()
    {
        Indexed.Register();
        using var native = fixture.Gpu.Allocate<int>(1);
        var view = native.DeviceView;
        int[] denominator = [0], nestedOutput = [43], output = [41];
        int nestedFailures = 0;
        var first = new Indexed([10])
        {
            CommitAction = () =>
            {
                Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1,
                    i => nestedOutput[i] = 7 / denominator[i]));
                nestedFailures++;
            }
        };
        var second = new Indexed([20])
        {
            CommitAction = () => fixture.Gpu.For(0, 1, i => view[i] += 5)
        };
        fixture.Gpu.For(0, 1, i =>
        {
            first[i]++;
            second[i] += 2;
            output[i] = first[i] + second[i];
        });
        Assert.Equal(new[] { 33 }, output);
        Assert.Equal(new[] { 11 }, first.Buffer);
        Assert.Equal(new[] { 22 }, second.Buffer);
        Assert.Equal(1, first.CommitCount);
        Assert.Equal(1, second.CommitCount);
        Assert.Equal(1, nestedFailures);
        Assert.Equal(new[] { 43 }, nestedOutput);
        Assert.Equal(new[] { 5 }, native.ToArray());

        fixture.Gpu.For(0, 1, i => view[i] += 2);
        Assert.Equal(new[] { 7 }, native.ToArray());
    }

    private sealed class Indexed(int[] values)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<Indexed, int>(value => value.Capture());
            return true;
        });
        public static void Register() => _ = registered.Value;
        public int[] Buffer { get; } = values;
        public Action? CaptureAction;
        public Action? CommitAction;
        public int CaptureCount;
        public int CommitCount;
        public int this[int index] { get => Buffer[index]; set => Buffer[index] = value; }

        private GpuCollectionSnapshot<int> Capture()
        {
            CaptureCount++;
            CaptureAction?.Invoke();
            return new(Buffer.ToArray(), commit: Commit, identity: this);
        }

        private void Commit(int[] updated)
        {
            updated.CopyTo(Buffer, 0);
            CommitCount++;
            CommitAction?.Invoke();
        }
    }
}
