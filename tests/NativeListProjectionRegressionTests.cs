using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class NativeListProjectionRegressionTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DetachedAdapterAndNativeListHelpersKeepIndependentStorage(bool nativeFirst)
    {
        DetachedList.Register();
        var custom = new DetachedList();
        List<int> native = custom;
        int[] observed = [-1, -1, -1, -1];
        if (nativeFirst)
            fixture.Gpu.For(0, 1, i =>
            {
                observed[0] = native[i];
                observed[1] = custom[i];
                NativeWrite(custom, i, NativeRead(custom, i) + 100);
                custom[i + 1] += 5;
                observed[2] = NativeCount(custom);
                observed[3] = custom.Count;
            });
        else
            fixture.Gpu.For(0, 1, i =>
            {
                observed[1] = custom[i];
                observed[0] = NativeRead(custom, i);
                NativeWrite(custom, i, NativeRead(custom, i) + 100);
                custom[i + 1] += 5;
                observed[2] = NativeCount(custom);
                observed[3] = custom.Count;
            });
        Assert.Equal(new[] { 7, 50, 3, 2 }, observed);
        Assert.Equal(new[] { 107, 8, 9 }, native);
        Assert.Equal(new[] { 50, 65 }, custom.Data);
        Assert.Equal(1, custom.CommitCount);
        Assert.Equal(0, custom.HostAccessorCalls);
    }

    [Fact]
    public void NativeProjectionSurvivesHelperReturnAndLocalReferenceAssignment()
    {
        DetachedList.Register();
        var custom = new DetachedList();
        int[] observed = [-1, -1, -1];
        fixture.Gpu.For(0, 1, i =>
        {
            List<int> local = ReturnNative(custom);
            observed[0] = local[i + 2];
            local[i + 1] += 100;
            observed[1] = ReturnNative(custom).Count;
            observed[2] = custom[i + 1];
        });
        Assert.Equal(new[] { 9, 3, 60 }, observed);
        Assert.Equal(new[] { 7, 108, 9 }, (List<int>)custom);
        Assert.Equal(new[] { 50, 60 }, custom.Data);
        Assert.Equal(0, custom.CommitCount);
        Assert.Equal(0, custom.HostAccessorCalls);
    }

    [Fact]
    public void NativeInterfaceHelpersUseWritableBaseStorageBesideReadOnlyDetachedAdapter()
    {
        DetachedList.Register();
        var custom = new DetachedList(readOnly: true);
        int[] observed = [-1, -1, -1, -1];
        fixture.Gpu.For(0, 1, i =>
        {
            observed[0] = custom[i];
            InterfaceWrite(custom, i, InterfaceRead(custom, i) + 100);
            observed[1] = InterfaceRead(custom, i);
            observed[2] = InterfaceCount(custom);
            observed[3] = custom[i];
        });
        Assert.Equal(new[] { 50, 107, 3, 50 }, observed);
        Assert.Equal(new[] { 107, 8, 9 }, (List<int>)custom);
        Assert.Equal(new[] { 50, 60 }, custom.Data);
        Assert.Equal(0, custom.CommitCount);
        Assert.Equal(0, custom.HostAccessorCalls);
    }

    [Fact]
    public void DetachedAdapterErrorRollsBackWritesToItsNativeProjection()
    {
        DetachedList.Register();
        var custom = new DetachedList(readOnly: true);
        int[] observed = [-1];
        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, 1, i =>
        {
            NativeWrite(custom, i, 100);
            observed[i] = 100;
            custom[i] = 100;
        }));
        Assert.Equal(new[] { 7, 8, 9 }, (List<int>)custom);
        Assert.Equal(new[] { 50, 60 }, custom.Data);
        Assert.Equal(new[] { -1 }, observed);
        Assert.Equal(0, custom.CommitCount);
        fixture.Gpu.For(0, 1, i => observed[i] = NativeRead(custom, i));
        Assert.Equal(new[] { 7 }, observed);
    }

    [Fact]
    public void EmptyDetachedAdapterCanStillExposeItsNonEmptyNativeList()
    {
        DetachedList.Register();
        var custom = new DetachedList(emptyCustom: true);
        int[] observed = [-1, -1, -1];
        fixture.Gpu.For(0, 1, i =>
        {
            observed[0] = custom.Count;
            observed[1] = NativeCount(custom);
            NativeWrite(custom, i + 2, 100);
            observed[2] = NativeRead(custom, i + 2);
        });
        Assert.Equal(new[] { 0, 3, 100 }, observed);
        Assert.Equal(new[] { 7, 8, 100 }, (List<int>)custom);
        Assert.Empty(custom.Data);
        Assert.Equal(0, custom.CommitCount);
    }

    [Fact]
    public void EmptyNativeListBoundsErrorRollsBackItsNonEmptyDetachedAdapter()
    {
        DetachedList.Register();
        var custom = new DetachedList(emptyNative: true);
        int[] observed = [-1, -1, -1];
        Assert.Throws<ArgumentOutOfRangeException>(() => fixture.Gpu.For(0, 1, i =>
        {
            custom[i] = 100;
            observed[0] = NativeCount(custom);
            NativeWrite(custom, i, 100);
        }));
        Assert.Empty((List<int>)custom);
        Assert.Equal(new[] { 50, 60 }, custom.Data);
        Assert.Equal(new[] { -1, -1, -1 }, observed);
        Assert.Equal(0, custom.CommitCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void CustomAdapterCannotBecomeDifferentNativeInterfaceIndexerThroughHelper(int scenario)
    {
        SplitInterfaceView.Register();
        var custom = new SplitInterfaceView();
        int[] observed = [-1];
        Action<int> body = scenario switch
        {
            0 => i => observed[i] = InterfaceRead(custom, i),
            1 => i => observed[i] = ReturnInterface(custom)[i],
            _ => i => observed[i] = ((IReadOnlyList<int>)custom)[i]
        };
        var error = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, body));
        Assert.Contains("indexer", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(new[] { -1 }, observed);
        Assert.Equal(new[] { 50, 60 }, custom.Data);
        Assert.Equal(new[] { 7, 8, 9 }, custom.InterfaceData);
        Assert.Equal(0, custom.CaptureCount);
        fixture.Gpu.For(0, 1, i => observed[i] = custom[i]);
        Assert.Equal(new[] { 50 }, observed);
    }

    [Fact]
    public void OptimizedNativeCountConversionRejectsDifferentInterfaceCount()
    {
        SplitCountView.Register();
        var custom = new SplitCountView();
        int[] observed = [-1];
        var error = Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1,
            i => observed[i] = ((IReadOnlyCollection<int>)custom).Count));
        Assert.Contains("indexer", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(new[] { -1 }, observed);
        Assert.Equal(0, custom.CaptureCount);
    }

    private static int NativeRead(List<int> values, int index) => values[index];
    private static void NativeWrite(List<int> values, int index, int value) => values[index] = value;
    private static int NativeCount(List<int> values) => values.Count;
    private static List<int> ReturnNative(DetachedList values) => values;
    private static int InterfaceRead(IReadOnlyList<int> values, int index) => values[index];
    private static void InterfaceWrite(IList<int> values, int index, int value) => values[index] = value;
    private static int InterfaceCount(IReadOnlyList<int> values) => values.Count;
    private static IReadOnlyList<int> ReturnInterface(SplitInterfaceView values) => values;

    private sealed class DetachedList : List<int>
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<DetachedList, int>(view =>
                new(view.Data, view.readOnly, _ => view.CommitCount++));
            return true;
        });
        private readonly bool readOnly;
        public int[] Data { get; } = [50, 60];
        public int CommitCount;
        public int HostAccessorCalls;
        public DetachedList(bool readOnly = false, bool emptyNative = false, bool emptyCustom = false)
            : base(emptyNative ? Array.Empty<int>() : [7, 8, 9])
        {
            this.readOnly = readOnly;
            if (emptyCustom)
                Data = [];
        }
        public new int Count { get { HostAccessorCalls++; return Data.Length; } }
        public new int this[int index]
        {
            get { HostAccessorCalls++; return Data[index]; }
            set { HostAccessorCalls++; Data[index] = value; }
        }
        public static void Register() => _ = registered.Value;
    }

    private sealed class SplitInterfaceView : IReadOnlyList<int>
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<SplitInterfaceView, int>(view =>
            {
                view.CaptureCount++;
                return new(view.Data);
            });
            return true;
        });
        public int[] Data { get; } = [50, 60];
        public int[] InterfaceData { get; } = [7, 8, 9];
        public int CaptureCount;
        public int Count => Data.Length;
        public int this[int index] => Data[index];
        int IReadOnlyCollection<int>.Count => InterfaceData.Length;
        int IReadOnlyList<int>.this[int index] => InterfaceData[index];
        public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)Data).GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
        public static void Register() => _ = registered.Value;
    }

    private sealed class SplitCountView : IReadOnlyCollection<int>
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<SplitCountView, int>(view =>
            {
                view.CaptureCount++;
                return new(view.Data);
            });
            return true;
        });
        public int[] Data { get; } = [50, 60];
        public int CaptureCount;
        public int Count => Data.Length;
        public int this[int index] => Data[index];
        int IReadOnlyCollection<int>.Count => 3;
        public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)Data).GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
        public static void Register() => _ = registered.Value;
    }
}
