using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AdapterFlowAuditRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void ParentAndChildAdapterJoinsKeepInheritedAccessorsAndSharedSlices()
    {
        FlowView.Register();
        var storage = new FlowStorage([10, 20, 30, 40]);
        IFlowParent? missing = null;
        IFlowChild child = new FlowView(storage, false);
        IFlowParent parent = new FlowView(storage, false);
        int[] observed = [-1, -1];
        fixture.Gpu.For(0, observed.Length, i =>
        {
            var selected = SelectView(i == 0, missing, parent, child);
            selected[i] += selected.Count;
            observed[i] = (missing ?? child)[i] + parent.Count;
        });
        Assert.Equal(new[] { 10, 22, 32, 40 }, storage.Buffer);
        Assert.Equal(new[] { 24, 34 }, observed);
        Assert.Equal(1, storage.CommitCount);
        Assert.Equal(0, storage.HostAccessorCalls);
    }

    [Fact]
    public void SelectedReadOnlyAdapterRetainsItsRestrictionAfterHelperReturn()
    {
        FlowView.Register();
        var storage = new FlowStorage([10, 20, 30, 40]);
        IFlowParent writable = new FlowView(storage, false);
        IFlowChild readOnly = new FlowView(storage, true);
        int[] observed = [-1, -1];
        fixture.Gpu.For(0, observed.Length, i =>
        {
            writable[i] += 3;
            observed[i] = SelectView(false, null, writable, readOnly)[i];
        });
        Assert.Equal(new[] { 10, 23, 33, 40 }, storage.Buffer);
        Assert.Equal(new[] { 23, 33 }, observed);
        Assert.Equal(1, storage.CommitCount);

        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, 1, i =>
        {
            writable[i] = 99;
            observed[i] = 99;
            SelectView(false, null, writable, readOnly)[i] = 123;
        }));
        Assert.Equal(new[] { 10, 23, 33, 40 }, storage.Buffer);
        Assert.Equal(new[] { 23, 33 }, observed);
        Assert.Equal(1, storage.CommitCount);
        Assert.Equal(0, storage.HostAccessorCalls);
    }

    [Fact]
    public void ReturnedMemorySlicesAndElementReferencesKeepAliasedStorage()
    {
        int[] owner = [10, 20, 30, 40];
        Memory<int> first = owner.AsMemory(0, 3);
        Memory<int> second = owner.AsMemory(1, 3);
        ReadOnlyMemory<int> readOnly = owner.AsMemory(1, 2);
        int[] observed = [-1, -1];
        fixture.Gpu.For(0, observed.Length, i =>
        {
            ref int element = ref Element(SelectMemory(i == 0, first, second), i);
            element += 5;
            observed[i] = ReadOnlyElement(readOnly, i);
        });
        Assert.Equal(new[] { 10, 25, 35, 40 }, owner);
        Assert.Equal(new[] { 25, 35 }, observed);
    }

    [Fact]
    public void ReturnedMemorySliceBoundsErrorRejectsWritesThroughOtherAliases()
    {
        int[] owner = [10, 20, 30, 40];
        Memory<int> first = owner.AsMemory(0, 3);
        Memory<int> second = owner.AsMemory(1, 3);
        int[] observed = [-1];
        Assert.Throws<IndexOutOfRangeException>(() => fixture.Gpu.For(0, 1, i =>
        {
            Element(first, i) = 99;
            observed[i] = Element(SelectMemory(false, first, second), 2);
        }));
        Assert.Equal(new[] { 10, 20, 30, 40 }, owner);
        Assert.Equal(new[] { -1 }, observed);
    }

    private static IFlowParent SelectView(bool selectParent, IFlowParent? missing,
        IFlowParent parent, IFlowChild child) => selectParent ? missing ?? parent : missing ?? child;
    private static Memory<int> SelectMemory(bool first, Memory<int> left, Memory<int> right) =>
        first ? left.Slice(1, 2) : right.Slice(0, 2);
    private static ref int Element(Memory<int> values, int index) => ref values.Span[index];
    private static ref readonly int ReadOnlyElement(ReadOnlyMemory<int> values, int index) => ref values.Span[index];

    private interface IFlowParent
    {
        FlowStorage Storage { get; }
        bool ReadOnly { get; }
        int Count { get; }
        int this[int index] { get; set; }
    }
    private interface IFlowChild : IFlowParent;

    private sealed class FlowStorage(int[] buffer)
    {
        public int[] Buffer { get; } = buffer;
        public int CommitCount;
        public int HostAccessorCalls;
        public void Commit(int[] values) => CommitCount++;
    }

    private sealed class FlowView(FlowStorage storage, bool readOnly) : IFlowChild
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IFlowParent, int>(Capture);
            GpuCollectionAdapters.Register<IFlowChild, int>(Capture);
            return true;
        });
        private static GpuCollectionSnapshot<int> Capture(IFlowParent view) =>
            new(view.Storage.Buffer, 1, 2, view.ReadOnly, view.Storage.Commit, view.Storage);
        public static void Register() => _ = registered.Value;
        public FlowStorage Storage => storage;
        public bool ReadOnly => readOnly;
        public int Count { get { storage.HostAccessorCalls++; return 2; } }
        public int this[int index]
        {
            get { storage.HostAccessorCalls++; return storage.Buffer[index + 1]; }
            set { storage.HostAccessorCalls++; storage.Buffer[index + 1] = value; }
        }
    }
}
