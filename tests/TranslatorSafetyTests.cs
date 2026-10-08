using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class TranslatorSafetyTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;
    private static readonly int[] StaticCollection = [17];
    private static int mutableStatic;

    [Fact]
    public void InlineConditionalReferenceComparisonsKeepTheirCollectionCategory()
    {
        int[] values = [2, 3];
        IReadOnlyList<int> same = values;
        IReadOnlyList<int> other = new List<int> { 2, 3 };
        int[] output = new int[4];
        Gpu.For<int>(0, output.Length, i =>
        {
            output[i] = (i % 2 == 0 ? same : values) == null ? -1 : 1;
            if ((i % 2 == 0 ? other : values) == values)
                output[i] += 10;
        });
        Assert.Equal(new[] { 1, 11, 1, 11 }, output);
    }

    [Fact]
    public void CapturedArraySegmentMayWriteBackingElementsButCannotChangeItsDescriptor()
    {
        int[] values = [10, 20, 30, 40];
        var segment = new ArraySegment<int>(values, 1, 2);
        int[] output = [41];
        Gpu.For<int>(0, segment.Count, i => segment[i] += 3);
        Assert.Equal(new[] { 10, 23, 33, 40 }, values);
        Assert.Equal(1, segment.Offset);
        Assert.Equal(2, segment.Count);

        Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            ref var alias = ref segment;
            alias = default;
        }));
        Assert.Equal(41, output[0]);
        Assert.Same(values, segment.Array);
        Assert.Equal(1, segment.Offset);
        Assert.Equal(2, segment.Count);
        RunRecoveryLoop(output);
    }

    [Fact]
    public void ClosureValueReferencesCannotEscapeThroughLocalsAndHelpers()
    {
        var captured = new MutableValue { Number = 7 };
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            ref var alias = ref captured;
            alias.Number++;
        }));
        Assert.Equal(7, captured.Number);
        Assert.Equal(41, output[0]);
        Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            MutateCaptured(ref captured);
        }));
        Assert.Equal(7, captured.Number);
        Assert.Equal(41, output[0]);
        RunRecoveryLoop(output);
    }

    [Fact]
    public void RejectedHelperAllocationDoesNotPoisonLaterKernelCompilation()
    {
        int[] input = [7];
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, _ => output[0] = AllocatingHelper(input)));
        Assert.Equal(41, output[0]);
        Assert.Equal(7, input[0]);
        RunRecoveryLoop(output);
    }

    [Fact]
    public void HelpersCannotCaptureTheirOwnStaticCollectionsImplicitly()
    {
        int[] input = [7];
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, _ => output[0] = HiddenCollectionHelper(input)));
        Assert.Equal(41, output[0]);
        RunRecoveryLoop(output);
    }

    [Fact]
    public void HelpersCannotMutateStaticStateOrPoisonFollowingLoops()
    {
        mutableStatic = 7;
        int[] input = [3];
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, _ => output[0] = MutatingStaticHelper(input)));
        Assert.Equal(7, mutableStatic);
        Assert.Equal(41, output[0]);
        RunRecoveryLoop(output);
    }

    [Fact]
    public void ReadOnlySnapshotAliasesKeepWritableAliasesAndRollbackRejectedWrites()
    {
        ReadOnlyIndexed.Initialize();
        int[] values = [2, 4];
        var readOnly = new ReadOnlyIndexed(values);
        int[] output = [41, 43];
        Gpu.For<int>(0, values.Length, i =>
        {
            values[i] += readOnly[i];
            output[i] = readOnly[i];
        });
        Assert.Equal(new[] { 4, 8 }, values);
        Assert.Equal(new[] { 4, 8 }, output);

        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, values.Length, i =>
        {
            output[i] = 99;
            values[i] += 3;
            readOnly[i] = 17;
        }));
        Assert.Equal(new[] { 4, 8 }, values);
        Assert.Equal(new[] { 4, 8 }, output);
        RunRecoveryLoop(output);
    }

    [Fact]
    public void CachedInterfaceCapturesCanChangeRuntimeCollectionKindAndLength()
    {
        IList<int>? values = new[] { 3, 5 };
        int[] output = new int[3];
        Action<int> body = _ =>
        {
            output[0] = values == null ? -1 : values.Count;
            output[1] = values == null ? -1 : values[0];
            output[2] = values != null && values.Count > 1 ? values[1] : -1;
        };
        Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { 2, 3, 5 }, output);
        values = new List<int> { 7 };
        Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { 1, 7, -1 }, output);
        values = null;
        Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { -1, -1, -1 }, output);
    }

    [Fact]
    public void CopiedClosureValuesAndMemoryDescriptorsRemainIndependent()
    {
        var captured = new MutableValue { Number = 7 };
        int[] values = [2, 3, 5];
        var segment = new ArraySegment<int>(values, 1, 2);
        int[] output = new int[2];
        Gpu.For<int>(0, 1, _ =>
        {
            var local = captured;
            local.Number += 11;
            var copiedSegment = segment;
            copiedSegment = copiedSegment.Slice(1, 1);
            copiedSegment[0] += local.Number;
            output[0] = local.Number;
            output[1] = copiedSegment.Count;
        });
        Assert.Equal(7, captured.Number);
        Assert.Equal(1, segment.Offset);
        Assert.Equal(2, segment.Count);
        Assert.Equal(new[] { 2, 3, 23 }, values);
        Assert.Equal(new[] { 18, 1 }, output);
    }

    [Fact]
    public void CachedBodyCanRecoverAfterADeferredBoundsFailure()
    {
        int[] values = [7, 11];
        int[] output = [41];
        int index = 2;
        Action<int> body = _ =>
        {
            output[0] = 99;
            values[index] += 3;
            output[0] = values[index];
        };
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, body));
        Assert.Equal(new[] { 7, 11 }, values);
        Assert.Equal(41, output[0]);
        index = 1;
        Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { 7, 14 }, values);
        Assert.Equal(14, output[0]);
    }

    [Fact]
    public void ConstructorStaticEffectsAreRejectedBeforeCompilationAndFollowingLoopsWork()
    {
        int[] input = [7];
        int[] output = [41];
        var failure = Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1,
            _ => output[0] = new NativeConstruction(input[0]).Number));
        Assert.Contains("static", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(41, output[0]);
        RunRecoveryLoop(output);
    }

    [Fact]
    public void PrimitiveOnlyHelpersCannotCacheMutableStaticReads()
    {
        mutableStatic = 7;
        int[] output = [41];
        var failure = Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1,
            i => output[i] = ReadMutableStatic(i)));
        Assert.Contains("static", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(41, output[0]);
        Assert.Equal(7, mutableStatic);
        RunRecoveryLoop(output);
    }

    [Fact]
    public void DeferredLongReferencesStayAlignedWhenLargestCapturedElementHasTwelveBytes()
    {
        long[] empty = [];
        TripleInt[] triples = new TripleInt[4];
        int[] output = [-1, -1, -1, -1];
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For(0, output.Length, i =>
        {
            StoreLong(ref empty[i]);
            output[i] = triples[i].A;
        }));
        Assert.Equal(new[] { -1, -1, -1, -1 }, output);
        Assert.Empty(empty);
        RunRecoveryLoop(output);
    }

    private static int ReadMutableStatic(int input) => input + mutableStatic;
    private static void StoreLong(ref long value) => value = long.MaxValue;

    public struct NativeConstruction
    {
        public int Number;
        public NativeConstruction(int input) => Number = input + StaticCollection[0];
    }

    public struct TripleInt
    {
        public int A, B, C;
    }

    private void RunRecoveryLoop(int[] output)
    {
        Gpu.For<int>(0, output.Length, i => output[i] = i + 11);
        Assert.Equal(Enumerable.Range(0, output.Length).Select(i => i + 11), output);
    }

    private static int AllocatingHelper(int[] input)
    {
        int[] allocated = new int[1];
        allocated[0] = input[0];
        return allocated[0];
    }

    private static int HiddenCollectionHelper(int[] input) => input[0] + StaticCollection[0];
    private static int MutatingStaticHelper(int[] input)
    {
        mutableStatic++;
        return input[0] + mutableStatic;
    }
    private static void MutateCaptured(ref MutableValue value) => value.Number++;
    private struct MutableValue { public int Number; }

    private sealed class ReadOnlyIndexed(int[] values)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<ReadOnlyIndexed, int>(collection => new(collection.Values, isReadOnly: true));
            return true;
        });
        private int[] Values => values;
        public int Count => values.Length;
        public int this[int index] { get => values[index]; set => values[index] = value; }
        internal static void Initialize() => _ = registered.Value;
    }
}
