using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class PerformanceGuardRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void PureMathAndNativeAccessorsKeepDirectAndTranslatedHelperResults()
    {
        double[] input = [0.25, 1, 2.25, 4];
        double[] direct = new double[input.Length], helper = new double[input.Length];
        using var buffer = fixture.Gpu.Upload(input);
        var view = buffer.DeviceView;
        fixture.Gpu.For(0, input.Length, i =>
        {
            double value = view[i];
            direct[i] = Math.Sqrt(Math.Abs(value)) + Math.CopySign(Math.Max(value, 0.5), -1) + Math.Round(value);
            helper[i] = PureMathHelper(value, view, i);
        });
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(Math.Sqrt(Math.Abs(input[i])) + Math.CopySign(Math.Max(input[i], 0.5), -1) + Math.Round(input[i]), direct[i]);
            Assert.Equal(Math.Round(Math.Sqrt(Math.Abs(input[i])) + Math.Min(input[i], 2)) +
                unchecked((sbyte)(i + 129)) + input.Length + sizeof(double) + input[i] % 0.5, helper[i]);
        }
    }

    [Fact]
    public void AFailedAutomaticReferenceStillStopsBeforeNativeWritesAfterPureMath()
    {
        int[] input = [], output = [41];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Action<int> body = i =>
        {
            output[i] = 99;
            ref int element = ref AddressAfterPureMath(input, view);
            element++;
            view[0]++;
            output[i] = element;
        };
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        Assert.Equal(0, visits.ToArray()[0]);

        input = [7];
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(8, input[0]);
        Assert.Equal(8, output[0]);
        Assert.Equal(1, visits.ToArray()[0]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void FailureCapableMathAndIntegerDivisionKeepTheirImmediateExits(int operation)
    {
        int[] arguments = [0, int.MinValue, 2, 1, 16], output = [41];
        double[] values = [double.NaN, 1.25];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Action<int> body = i =>
        {
            output[i] = 99;
            int result = (int)Math.Round(Math.Sqrt(Math.Abs(values[1])));
            switch (operation)
            {
                case 0: result += 8 / arguments[0]; break;
                case 1: result += Math.Abs(arguments[1]); break;
                case 2: result += Math.Clamp(1, arguments[2], arguments[3]); break;
                case 3: result += (int)Math.Round(values[1], arguments[4]); break;
                case 4: result += Math.Sign(values[0]); break;
                default: result += Math.DivRem(8, arguments[0], out int remainder) + remainder; break;
            }
            view[0]++;
            output[i] = result;
        };
        switch (operation)
        {
            case 0 or 5: Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, body)); break;
            case 1: Assert.Throws<OverflowException>(() => fixture.Gpu.For(0, 1, body)); break;
            case 2: Assert.Throws<ArgumentException>(() => fixture.Gpu.For(0, 1, body)); break;
            case 3: Assert.Throws<ArgumentOutOfRangeException>(() => fixture.Gpu.For(0, 1, body)); break;
            default: Assert.Throws<ArithmeticException>(() => fixture.Gpu.For(0, 1, body)); break;
        }
        Assert.Equal(41, output[0]);
        Assert.Equal(0, visits.ToArray()[0]);

        arguments[0] = 2;
        arguments[1] = 17;
        arguments[2] = 0;
        arguments[3] = 2;
        arguments[4] = 1;
        values[0] = 1;
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(operation switch { 0 or 5 => 5, 1 => 18, _ => 2 }, output[0]);
        Assert.Equal(1, visits.ToArray()[0]);
    }

    [Fact]
    public void ReusedErrorStorageResetsBetweenDifferentFailuresAndNativeOnlyBodies()
    {
        int[] empty = [], denominator = [0], output = [41];
        int? optional = null;
        using var native = fixture.Gpu.Upload(new[] { 7 });
        var view = native.DeviceView;

        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, i => output[i] = empty[i]));
        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, i => output[i] = 8 / denominator[i]));
        fixture.Gpu.For(0, 1, i => view[i] += 3);
        Assert.Equal(new[] { 10 }, native.ToArray());
        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, 1, i => output[i] = optional!.Value));
        Assert.Equal(new[] { 41 }, output);

        fixture.Gpu.For(0, 1, i => output[i] = view[i] + 2);
        Assert.Equal(new[] { 12 }, output);
        fixture.Gpu.For(0, 1, i => view[i] += 5);
        Assert.Equal(new[] { 15 }, native.ToArray());
    }

    [Fact]
    public async Task ConcurrentHostCallsKeepEachInvocationErrorAndSuccessSeparate()
    {
        int[] empty = [], denominator = [0], boundsOutput = [41], divisionOutput = [43];
        using var native = fixture.Gpu.Upload(new[] { 0 });
        var view = native.DeviceView;
        Action<int> boundsBody = i => boundsOutput[i] = empty[i];
        Action<int> divisionBody = i => divisionOutput[i] = 8 / denominator[i];
        Action<int> successBody = i => view[i] += 3;
        using var start = new ManualResetEventSlim();
        var bounds = Task.Run(() =>
        {
            start.Wait();
            for (int repeat = 0; repeat < 4; repeat++)
                Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, boundsBody));
        });
        var division = Task.Run(() =>
        {
            start.Wait();
            for (int repeat = 0; repeat < 4; repeat++)
                Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, divisionBody));
        });
        var success = Task.Run(() =>
        {
            start.Wait();
            for (int repeat = 0; repeat < 4; repeat++)
                fixture.Gpu.For(0, 1, successBody);
        });
        start.Set();
        await Task.WhenAll(bounds, division, success);
        Assert.Equal(new[] { 41 }, boundsOutput);
        Assert.Equal(new[] { 43 }, divisionOutput);
        Assert.Equal(new[] { 12 }, native.ToArray());

        fixture.Gpu.For(0, 1, i => boundsOutput[i] = view[i] + 1);
        Assert.Equal(new[] { 13 }, boundsOutput);
    }

    [Fact]
    public void DisposingAnExecutorAfterErrorStorageReuseReleasesOwnedBuffers()
    {
        using var owner = new Gpu(new() { Backend = fixture.Gpu.Backend });
        using var native = owner.Upload(new[] { 7 });
        var view = native.DeviceView;
        int[] denominator = [0], output = [41];
        Assert.Throws<DivideByZeroException>(() => owner.For(0, 1, i => output[i] = 8 / denominator[i]));
        Action<int> successBody = i => view[i] += 3;
        owner.For(0, 1, successBody);
        Assert.Equal(new[] { 10 }, native.ToArray());

        owner.Dispose();
        owner.Dispose();
        Assert.Throws<ObjectDisposedException>(() => native.ToArray());
        Assert.Throws<ObjectDisposedException>(() => owner.For(0, 1, successBody));
        Assert.Throws<ObjectDisposedException>(() => owner.Allocate<int>(1));
    }

    [Fact]
    public void AdapterCaptureCanCatchANestedFailureBeforeTheOuterInvocationResetsErrors()
    {
        ReentrantIndexed.Register();
        var collection = new ReentrantIndexed(fixture.Gpu, [3, 5], failDuringCapture: true);
        int[] output = [-1, -1];
        fixture.Gpu.For(0, output.Length, i =>
        {
            collection[i] += 10;
            output[i] = collection[i];
        });
        Assert.Equal(new[] { 13, 15 }, collection.Buffer);
        Assert.Equal(new[] { 13, 15 }, output);
        Assert.Equal(1, collection.CaptureCount);
        Assert.Equal(1, collection.CommitCount);
        Assert.Equal(1, collection.NestedFailures);
        Assert.Equal(new[] { 41 }, collection.NestedOutput);
    }

    [Fact]
    public void AdapterCommitCanCatchANestedFailureAndTheNextBodyStartsWithClearErrors()
    {
        ReentrantIndexed.Register();
        var collection = new ReentrantIndexed(fixture.Gpu, [7, 11], failDuringCommit: true);
        int[] output = [-1, -1];
        fixture.Gpu.For(0, output.Length, i =>
        {
            collection[i] += 2;
            output[i] = collection[i];
        });
        Assert.Equal(new[] { 9, 13 }, collection.Buffer);
        Assert.Equal(new[] { 9, 13 }, output);
        Assert.Equal(1, collection.CaptureCount);
        Assert.Equal(1, collection.CommitCount);
        Assert.Equal(1, collection.NestedFailures);
        Assert.Equal(new[] { 41 }, collection.NestedOutput);

        using var native = fixture.Gpu.Upload(new[] { 1 });
        var view = native.DeviceView;
        fixture.Gpu.For(0, 1, i => view[i] += 5);
        Assert.Equal(new[] { 6 }, native.ToArray());
    }

    private static double PureMathHelper(double value, GpuView<double> source, int index) =>
        Math.Round(Math.Sqrt(Math.Abs(source[index])) + Math.Min(value, 2)) +
        unchecked((sbyte)(index + 129)) + source.Length + sizeof(double) + value % 0.5;

    private static ref int AddressAfterPureMath(int[] input, GpuView<int> visits)
    {
        _ = Math.Round(Math.Sqrt(Math.Abs((double)visits.Length)));
        return ref input[0];
    }

    private sealed class ReentrantIndexed(Gpu gpu, int[] values,
        bool failDuringCapture = false, bool failDuringCommit = false)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<ReentrantIndexed, int>(value => value.Capture());
            return true;
        });
        public static void Register() => _ = registered.Value;
        public int[] Buffer { get; } = values;
        public int[] NestedOutput { get; } = [41];
        public int CaptureCount;
        public int CommitCount;
        public int NestedFailures;
        public int this[int index] { get => Buffer[index]; set => Buffer[index] = value; }

        private GpuCollectionSnapshot<int> Capture()
        {
            CaptureCount++;
            if (failDuringCapture)
            {
                int[] empty = [];
                var nestedOutput = NestedOutput;
                Assert.Throws<IndexOutOfRangeException>(() => gpu.For(0, 1, i => nestedOutput[i] = empty[i]));
                NestedFailures++;
            }
            return new(Buffer.ToArray(), commit: Commit, identity: this);
        }

        private void Commit(int[] updated)
        {
            updated.CopyTo(Buffer, 0);
            CommitCount++;
            if (failDuringCommit)
            {
                int[] denominator = [0];
                var nestedOutput = NestedOutput;
                Assert.Throws<DivideByZeroException>(() => gpu.For(0, 1, i => nestedOutput[i] = 8 / denominator[i]));
                NestedFailures++;
            }
        }
    }
}
