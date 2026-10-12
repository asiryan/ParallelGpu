using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ReadOnlyFlowAuditRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void PureInstanceMethodsCanReassignConditionalReferencesBetweenIndependentCopies()
    {
        var captured = new Value { Number = 7 };
        int[] output = new int[4];
        fixture.Gpu.For(0, output.Length,
            i => output[i] = captured.TransformCopies(i % 2 == 0, i + 1));
        Assert.Equal(Enumerable.Range(0, output.Length)
            .Select(i => captured.TransformCopies(i % 2 == 0, i + 1)), output);
        Assert.Equal(7, captured.Number);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReceiverReferencesSelectedAfterLoopBackedgesAreRejectedBeforeHostWrites(bool useReceiver)
    {
        var captured = new Value { Number = 7 };
        int[] output = [41];
        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = captured.WriteAfterLoop(useReceiver);
        }));
        Assert.Contains("Captured values are read-only", failure.Message);
        Assert.Equal(7, captured.Number);
        Assert.Equal(41, output[0]);
        fixture.Gpu.For(0, 1, i => output[i] = captured.Number + 1);
        Assert.Equal(8, output[0]);
    }

    [Fact]
    public void ConditionalDescriptorReferencesCannotMutateCapturedArraySegments()
    {
        int[] values = [5, 7];
        var captured = new ArraySegment<int>(values, 1, 1);
        int[] output = [41, 43];
        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, output.Length, i =>
        {
            output[i] = 99;
            var local = captured;
            ref ArraySegment<int> selected = ref (i == 0 ? ref captured : ref local);
            selected = default;
        }));
        Assert.Contains("Captured values are read-only", failure.Message);
        Assert.Equal(new[] { 41, 43 }, output);
        Assert.Equal(new[] { 5, 7 }, values);
        Assert.Equal(1, captured.Offset);
        Assert.Equal(1, captured.Count);
        Assert.Same(values, captured.Array);
    }

    [Fact]
    public void NativeViewElementReferencesRemainWritableAcrossConditionalLocalsAndHelpers()
    {
        using var buffer = fixture.Gpu.Upload(new[] { 11, 13 });
        var captured = buffer.DeviceView;
        fixture.Gpu.For(0, 2, i =>
        {
            ref int selected = ref (i == 0 ? ref captured[0] : ref captured[1]);
            Increment(ref selected, i + 1);
        });
        Assert.Equal(new[] { 12, 15 }, buffer.ToArray());
        Assert.Equal(2, captured.Length);
    }

    [Fact]
    public void NativeViewDescriptorReferencesCannotEscapeToResetHelpers()
    {
        using var buffer = fixture.Gpu.Upload(new[] { 17 });
        var captured = buffer.DeviceView;
        int[] output = [41];
        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            Reset(ref captured);
        }));
        Assert.Contains("Captured values are read-only", failure.Message);
        Assert.Equal(41, output[0]);
        Assert.Equal(new[] { 17 }, buffer.ToArray());
        fixture.Gpu.For(0, 1, i => output[i] = captured[i]);
        Assert.Equal(17, output[0]);
    }

    private static void Increment(ref int value, int amount) => value += amount;
    private static void Reset(ref GpuView<int> value) => value = default;

    private struct Value
    {
        public int Number;

        public int TransformCopies(bool first, int count)
        {
            Value left = this, right = this;
            ref Value selected = ref (first ? ref left : ref right);
            for (int i = 0; i < count; i++)
            {
                Increment(ref selected.Number, i + 1);
                selected = ref (i % 2 == 0 ? ref right : ref left);
            }
            return left.Number + right.Number + Number;
        }

        public int WriteAfterLoop(bool useReceiver)
        {
            Value copy = this;
            ref Value selected = ref copy;
            for (int i = 0; i < 2; i++)
                if (i == 1 && useReceiver)
                    selected = ref this;
            selected.Number++;
            return selected.Number;
        }
    }
}
