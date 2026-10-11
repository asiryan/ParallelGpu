using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class DeepCaptureRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void PureInstanceMethodsCanMutateIndependentCopiesAndLocals()
    {
        var captured = new Entry(7);
        Entry[] output = new Entry[5];
        fixture.Gpu.For(0, output.Length, i =>
        {
            if (i == 0) output[i] = captured.Shift();
            if (i == 1) output[i] = captured.CreateDefault(true);
            if (i == 2) output[i] = captured.CreateDefault(false);
            if (i == 3) output[i] = captured.CreateConstructed();
            if (i == 4) output[i] = captured.ShiftRepeatedly(3);
        });
        Assert.Equal(new[] { 8, 9, 10, 11, 10 }, output.Select(value => value.Number));
        Assert.Equal(new[] { 70, 0, 0, 110, 70 }, output.Select(value => value.Nested.Number));
        Assert.Equal(7, captured.Number);
        Assert.Equal(70, captured.Nested.Number);
    }

    [Fact]
    public void PureInstanceCallsAndHelpersCanReadReceiverAndMutateLocalCopies()
    {
        var captured = new Entry(7);
        Entry[] output = new Entry[3];
        fixture.Gpu.For(0, output.Length, i =>
        {
            if (i == 0) output[i] = captured.ShiftViaInstance();
            if (i == 1) output[i] = captured.ShiftViaLocalHelper();
            if (i == 2) output[i] = captured.ShiftViaCopyTo();
        });
        Assert.All(output, value => Assert.Equal(8, value.Number));
        Assert.All(output, value => Assert.Equal(70, value.Nested.Number));
        Assert.Equal(7, captured.Number);
        Assert.Equal(70, captured.Nested.Number);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    public void ReceiverWritesAliasesAndRefHelpersAreRejectedBeforeAnyHostWrite(int operation)
    {
        var captured = new Entry(7);
        int[] output = [41];
        Action<int> body = operation switch
        {
            0 => _ => { output[0] = 99; captured.Mutate(); },
            1 => _ => { output[0] = 99; captured.MutateNested(); },
            2 => _ => { output[0] = 99; captured.Reset(); },
            3 => _ => { output[0] = 99; captured.Replace(); },
            4 => _ => { output[0] = 99; captured.MutateAlias(); },
            5 => _ => { output[0] = 99; captured.MutateFieldAlias(); },
            6 => _ => { output[0] = captured.ReadViaRefHelper(); },
            7 => _ => { output[0] = 99; captured.MutateConditionalAlias(true); },
            _ => _ => { output[0] = 99; captured.MutateConditionalAlias(false); },
        };
        var failure = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Contains("Captured values are read-only", failure.Message, StringComparison.Ordinal);
        Assert.Equal(7, captured.Number);
        Assert.Equal(70, captured.Nested.Number);
        Assert.Equal(41, output[0]);
        fixture.Gpu.For(0, 1, i => output[i] = captured.Shift().Number);
        Assert.Equal(8, output[0]);
    }

    private struct Entry
    {
        public int Number;
        public NestedEntry Nested;

        public Entry(int number)
        {
            Number = number;
            Nested = new NestedEntry { Number = number * 10 };
        }

        public Entry Shift()
        {
            Entry copy = this;
            copy.Number++;
            return copy;
        }

        public Entry CreateDefault(bool firstBranch)
        {
            Entry value = default;
            if (firstBranch) value.Number = Number + 2;
            else value.Number = Number + 3;
            return value;
        }

        public Entry CreateConstructed() => new(Number + 4);

        public Entry ShiftRepeatedly(int count)
        {
            Entry copy = this;
            for (int i = 0; i < count; i++) copy.Number++;
            return copy;
        }

        public Entry ShiftViaInstance() => Shift();

        public Entry ShiftViaLocalHelper()
        {
            Entry copy = this;
            Increment(ref copy.Number);
            return copy;
        }

        public Entry ShiftViaCopyTo()
        {
            Entry copy = default;
            CopyTo(ref copy);
            copy.Number++;
            return copy;
        }

        private void CopyTo(ref Entry destination) => destination = this;
        public void Mutate() => Number++;
        public void MutateNested() => Nested.Mutate();
        public void Reset() => this = default;
        public void Replace() => this = new Entry(99);

        public void MutateAlias()
        {
            ref Entry alias = ref this;
            alias.Number++;
        }

        public void MutateFieldAlias()
        {
            ref int alias = ref Number;
            alias++;
        }

        public int ReadViaRefHelper() => Read(ref this);

        public void MutateConditionalAlias(bool useReceiver)
        {
            Entry copy = this;
            ref Entry alias = ref (useReceiver ? ref this : ref copy);
            alias.Number++;
        }

        private static int Read(ref Entry value) => value.Number;
        private static void Increment(ref int value) => value++;
    }

    private struct NestedEntry
    {
        public int Number;
        public void Mutate() => Number++;
    }
}
