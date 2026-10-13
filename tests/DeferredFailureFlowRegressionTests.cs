using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class DeferredFailureFlowRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void InvalidReadsStopLoopBackedgesBeforeFallbackValuesKeepTheLoopRunning()
    {
        int[] input = [], output = [41];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            // The bound makes a broken implementation fail an assertion instead of hanging.
            while (input[0] == 0 && view[0] < 10_000)
            {
                output[0] = 99;
                view[0]++;
            }
        }));
        Assert.Equal(41, output[0]);
        Assert.InRange(visits.ToArray()[0], 0, 1);
        VerifyFollowingLoop(output);
    }

    [Fact]
    public void DivisionFailuresStopDoWhileBackedgesBeforeZeroFallbackKeepsTheLoopRunning()
    {
        int[] numerator = [7], denominator = [0], output = [41];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            do
            {
                output[0] = 99;
                view[0]++;
            }
            while (numerator[0] / denominator[0] == 0 && view[0] < 10_000);
        }));
        Assert.Equal(41, output[0]);
        Assert.Equal(1, visits.ToArray()[0]);
        VerifyFollowingLoop(output);
    }

    [Fact]
    public void InvalidScratchReferenceReadsStopBackedgesInsideTranslatedHelpers()
    {
        long[] input = [];
        int[] output = [41];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = ReadUntilNonzero(input, view);
        }));
        Assert.Equal(41, output[0]);
        Assert.InRange(visits.ToArray()[0], 0, 1);
        VerifyFollowingLoop(output);
    }

    [Fact]
    public void ArithmeticOnlyHelperBackedgesStopAfterReceivingAFailedReadFallback()
    {
        int[] input = [], output = [41];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = CountUntilNonzero(input[0], view);
        }));
        Assert.Equal(41, output[0]);
        Assert.InRange(visits.ToArray()[0], 0, 1);
        VerifyFollowingLoop(output);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LargeScratchAliasesRetainFirstFailureAfterLaterInvalidArithmetic(bool readOnly)
    {
        ScratchPayload[] input = [];
        int[] output = [41];
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            if (readOnly)
                output[0] = ReadMissing(input);
            else
                output[0] = UpdateMissing(input);
        }));
        Assert.Empty(input);
        Assert.Equal(41, output[0]);
        VerifyFollowingLoop(output);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void InvalidMathArgumentsRejectAllWritesAndCachedBodiesRecover(int operation)
    {
        double[] input = [1.25], output = [41];
        int[] digits = [16], modes = [int.MaxValue], bounds = [2, 1];
        int[] journal = [43];
        Action<int> body = i =>
        {
            journal[i] = 99;
            output[i] = operation == 0 ? Math.Round(input[i], digits[i]) :
                operation == 1 ? Math.Round(input[i], (MidpointRounding)modes[i]) :
                Math.Clamp(input[i], bounds[0], bounds[1]);
        };
        if (operation == 0)
            Assert.Throws<ArgumentOutOfRangeException>(() => fixture.Gpu.For(0, 1, body));
        else
            Assert.Throws<ArgumentException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        Assert.Equal(43, journal[0]);
        digits[0] = 1;
        modes[0] = (int)MidpointRounding.AwayFromZero;
        bounds[0] = 0;
        bounds[1] = 2;
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(operation == 0 ? Math.Round(input[0], digits[0]) :
            operation == 1 ? Math.Round(input[0], (MidpointRounding)modes[0]) :
            Math.Clamp(input[0], bounds[0], bounds[1]), output[0]);
        Assert.Equal(99, journal[0]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void ArithmeticFailuresReturnSafelyFromEveryMappedHelperResultAndCachedCallsRecover(int operation)
    {
        ScratchPayload[] payloads = [new() { First = 17, Last = 23 }];
        int[] elements = [19], denominator = [0], output = [41];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Action<int> body = i =>
        {
            output[i] = 99;
            switch (operation)
            {
                case 0:
                    var result = ValueAfterDivision(denominator[i], view);
                    output[i] = result.Number + (result.Enabled ? result.Marker : 0);
                    break;
                case 1:
                    WriteAfterDivision(output, denominator[i], view);
                    break;
                case 2:
                    ref ScratchPayload payload = ref PayloadAfterDivision(payloads, denominator[i], view);
                    payload.First++;
                    output[i] = (int)payload.First;
                    break;
                case 3:
                    ref int element = ref ElementAfterDivision(elements, denominator[i], view);
                    element++;
                    output[i] = element;
                    break;
                case 4:
                    int[] selected = elements;
                    ref int[] descriptor = ref DescriptorAfterDivision(ref selected, denominator[i], view);
                    descriptor[0]++;
                    output[i] = descriptor[0];
                    break;
                default:
                    var span = SpanAfterDivision(payloads, denominator[i], view);
                    span[0].Last++;
                    output[i] = (int)span[0].Last;
                    break;
            }
        };

        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        Assert.Equal(17, payloads[0].First);
        Assert.Equal(23, payloads[0].Last);
        Assert.Equal(19, elements[0]);
        Assert.InRange(visits.ToArray()[0], 0, 1);

        denominator[0] = 2;
        visits.CopyFrom(new[] { 0 });
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(operation switch { 0 => 68, 1 => 3, 2 => 18, 3 or 4 => 20, _ => 24 }, output[0]);
        Assert.Equal(1, visits.ToArray()[0]);
        Assert.Equal(operation == 2 ? 18 : 17, payloads[0].First);
        Assert.Equal(operation == 5 ? 24 : 23, payloads[0].Last);
        Assert.Equal(operation is 3 or 4 ? 20 : 19, elements[0]);
        VerifyFollowingLoop(output);
    }

    [Fact]
    public void NestedNumericHelperFailuresStopItsCallersLoopWithoutCollectionParameters()
    {
        int[] denominator = [0], output = [41];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Action<int> body = i =>
        {
            output[i] = 99;
            output[i] = CountByNestedDivision(denominator[i], view);
        };
        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        Assert.InRange(visits.ToArray()[0], 0, 1);

        denominator[0] = 2;
        visits.CopyFrom(new[] { 0 });
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(3, output[0]);
        Assert.Equal(0, visits.ToArray()[0]);
        VerifyFollowingLoop(output);
    }

    [Fact]
    public void RefFieldHelpersWithoutArrayElementAddressesReserveSafeFailureScratch()
    {
        int[] denominator = [0], output = [41];
        using var visits = fixture.Gpu.Upload(new[] { 0 });
        var view = visits.DeviceView;
        Action<int> body = i =>
        {
            output[i] = 99;
            var value = new GuardedResult { Number = 17, Enabled = true, Marker = 'A' };
            ref int field = ref FieldAfterDivision(ref value, denominator[i], view);
            field++;
            output[i] = value.Number;
        };
        Assert.Throws<DivideByZeroException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Equal(41, output[0]);
        Assert.InRange(visits.ToArray()[0], 0, 1);

        denominator[0] = 2;
        visits.CopyFrom(new[] { 0 });
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(21, output[0]);
        Assert.Equal(1, visits.ToArray()[0]);
        VerifyFollowingLoop(output);
    }

    private static int ReadUntilNonzero(long[] input, GpuView<int> visits)
    {
        ref long missing = ref RefMissing(input);
        while (missing == 0 && visits[0] < 10_000)
            visits[0]++;
        return visits[0];
    }

    private static ref long RefMissing(long[] input) => ref input[0];
    private static int CountUntilNonzero(int value, GpuView<int> visits)
    {
        while (value == 0 && visits[0] < 10_000)
            visits[0]++;
        return visits[0];
    }
    private static int ReadMissing(ScratchPayload[] input)
    {
        ReadOnlySpan<ScratchPayload> values = input;
        ref readonly ScratchPayload value = ref values[0];
        return 7 / (int)value.Last;
    }

    private static int UpdateMissing(ScratchPayload[] input)
    {
        ref ScratchPayload value = ref input.AsSpan()[0];
        value.First = long.MaxValue;
        value.B = value.C = value.D = value.E = value.F = value.G = long.MinValue;
        value.Last = 0;
        return 7 / (int)value.Last;
    }

    private static GuardedResult ValueAfterDivision(int denominator, GpuView<int> visits)
    {
        int number = NestedDivision(denominator);
        visits[0]++;
        return new GuardedResult { Number = number, Enabled = true, Marker = 'A' };
    }

    private static void WriteAfterDivision(int[] output, int denominator, GpuView<int> visits)
    {
        int number = NestedDivision(denominator);
        visits[0]++;
        output[0] = number;
    }

    private static ref ScratchPayload PayloadAfterDivision(ScratchPayload[] values, int denominator, GpuView<int> visits)
    {
        int number = NestedDivision(denominator);
        visits[0] += number != 0 ? 1 : 0;
        return ref values[0];
    }

    private static ref int ElementAfterDivision(int[] values, int denominator, GpuView<int> visits)
    {
        int number = NestedDivision(denominator);
        visits[0] += number != 0 ? 1 : 0;
        return ref values[0];
    }

    private static ref int[] DescriptorAfterDivision(ref int[] values, int denominator, GpuView<int> visits)
    {
        int number = NestedDivision(denominator);
        visits[0] += number != 0 ? 1 : 0;
        return ref values;
    }

    private static Span<ScratchPayload> SpanAfterDivision(ScratchPayload[] values, int denominator, GpuView<int> visits)
    {
        int number = NestedDivision(denominator);
        visits[0] += number != 0 ? 1 : 0;
        return values;
    }

    private static int CountByNestedDivision(int denominator, GpuView<int> visits)
    {
        int value;
        while ((value = NestedDivision(denominator)) == 0 && visits[0] < 10_000)
            visits[0]++;
        return value;
    }

    private static int NestedDivision(int denominator) => NumericDivision(7, denominator);
    private static int NumericDivision(int numerator, int denominator) => numerator / denominator;
    private static ref int FieldAfterDivision(ref GuardedResult value, int denominator, GpuView<int> visits)
    {
        int number = NestedDivision(denominator);
        visits[0]++;
        value.Number += number;
        return ref value.Number;
    }

    private void VerifyFollowingLoop(int[] output)
    {
        fixture.Gpu.For(0, output.Length, i => output[i] = i + 17);
        Assert.Equal(17, output[0]);
    }

    private struct ScratchPayload
    {
        public long First, B, C, D, E, F, G, Last;
    }

    private struct GuardedResult
    {
        public int Number;
        public bool Enabled;
        public char Marker;
    }
}
