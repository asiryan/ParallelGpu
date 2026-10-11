using System.Runtime.CompilerServices;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AdapterRegressionFixTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RegisteredListSubclassKeepsItsSliceAndSharesNativeStorage(bool nativeFirst)
    {
        OffsetList.Register();
        var custom = new OffsetList([10, 20, 30]);
        List<int> native = custom;
        int[] observed = [-1, -1];
        int[] counts = [-1, -1];
        if (nativeFirst)
            Gpu.For<int>(0, 2, i =>
            {
                int nativeCount = native.Count;
                WriteOffset(custom, i, ReadOffset(custom, i) + 5);
                observed[i] = ReadNative(native, i + 1);
                if (i == 0)
                {
                    counts[0] = nativeCount;
                    counts[1] = custom.Count;
                }
            });
        else
            Gpu.For<int>(0, 2, i =>
            {
                custom[i] += 5;
                observed[i] = native[i + 1];
                if (i == 0)
                {
                    counts[0] = native.Count;
                    counts[1] = custom.Count;
                }
            });
        Assert.Equal(new[] { 10, 25, 35 }, native);
        Assert.Equal(new[] { 25, 35 }, observed);
        Assert.Equal(new[] { 3, 2 }, counts);
        Assert.Equal(1, custom.CaptureCount);
        Assert.Equal(1, custom.CommitCount);
        Assert.Equal(native, custom.CommittedValues);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RegisteredInterfaceOverListSubclassKeepsReadOnlyContractAndAliases(bool nativeFirst)
    {
        NativeOffsetList.Register();
        var owner = new NativeOffsetList([10, 20, 30]);
        IOffsetView custom = owner;
        List<int> native = owner;
        int[] observed = [-1, -1];
        if (nativeFirst)
            Gpu.For<int>(0, 2, i =>
            {
                native[i + 1] += 5;
                observed[i] = ReadView(custom, i);
            });
        else
            Gpu.For<int>(0, 2, i =>
            {
                int previous = custom[i];
                native[i + 1] = previous + 5;
                observed[i] = custom[i];
            });
        Assert.Equal(new[] { 10, 25, 35 }, native);
        Assert.Equal(new[] { 25, 35 }, observed);
        Assert.Equal(1, owner.CaptureCount);
        Assert.Equal(1, owner.CommitCount);
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, i =>
        {
            native[i + 1] = 99;
            observed[i] = 99;
            WriteView(custom, i, 101);
        }));
        Assert.Equal(new[] { 10, 25, 35 }, native);
        Assert.Equal(new[] { 25, 35 }, observed);
        Assert.Equal(1, owner.CommitCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ConcreteFloatAdapterDoesNotHideIntInterfaceAdapter(bool floatsFirst)
    {
        DualStorage.Register();
        var floats = new DualStorage();
        IIntView ints = floats;
        int[] observedInts = [-1];
        float[] observedFloats = [-1];
        if (floatsFirst)
            Gpu.For(0, 1, i =>
            {
                observedFloats[i] = ReadFloat(floats, i);
                floats[i] = 2.5f;
                ints[i] += 10;
                observedInts[i] = ReadInt(ints, i);
            });
        else
            Gpu.For(0, 1, i =>
            {
                ints[i] += 10;
                observedInts[i] = ints[i];
                observedFloats[i] = floats[i];
                floats[i] = 2.5f;
            });
        Assert.Equal(new[] { 17 }, ints.Buffer);
        Assert.Equal(new[] { 2.5f }, floats.Buffer);
        Assert.Equal(new[] { 17 }, observedInts);
        Assert.Equal(new[] { 1.5f }, observedFloats);
    }

    [Fact]
    public void AdapterUsesInheritedClassIndexerInDirectCallsAndHelpers()
    {
        InheritedView.Register();
        var input = new InheritedView([10, 20, 30]);
        int[] observed = [-1, -1];
        Gpu.For<int>(0, 2, i =>
        {
            input[i] += 5;
            observed[i] = ReadInherited(input, i) + input.Count;
        });
        Assert.Equal(new[] { 10, 25, 35 }, input.Buffer);
        Assert.Equal(new[] { 27, 37 }, observed);
        Assert.Equal(0, input.HostIndexerCalls);
    }

    [Fact]
    public void AdapterUsesParentInterfaceIndexerInDirectCallsAndHelpers()
    {
        ParentInterfaceView.Register();
        IChildView input = new ParentInterfaceView([10, 20, 30]);
        int[] observed = [-1, -1];
        Gpu.For<int>(0, 2, i =>
        {
            input[i] += 5;
            observed[i] = ReadChild(input, i) + input.Count;
        });
        var owner = Assert.IsType<ParentInterfaceView>(input);
        Assert.Equal(new[] { 10, 25, 35 }, owner.Buffer);
        Assert.Equal(new[] { 27, 37 }, observed);
        Assert.Equal(0, owner.HostIndexerCalls);
    }

    [Fact]
    public void AdapterUsesCustomIndexerMetadataForReadsAndWrites()
    {
        NamedView.Register();
        var input = new NamedView([10, 20, 30]);
        int[] observed = [-1, -1];
        Gpu.For<int>(0, 2, i =>
        {
            WriteNamed(input, i, ReadNamed(input, i) + 5);
            observed[i] = input[i] + input.Length;
        });
        Assert.Equal(new[] { 10, 25, 35 }, input.Buffer);
        Assert.Equal(new[] { 27, 37 }, observed);
        Assert.Equal(0, input.HostIndexerCalls);
    }

    [Fact]
    public void NamedValueAdapterAllowsElementWritesWithoutExecutingItsMutableHostIndexer()
    {
        NamedValueView.Register();
        var input = new NamedValueView([10, 20, 30]);
        int[] observed = [-1, -1];
        Gpu.For<int>(0, 2, i =>
        {
            input[i] += 5;
            observed[i] = ReadNamedValue(input, i) + input.Count;
        });
        Assert.Equal(new[] { 10, 25, 35 }, input.Buffer);
        Assert.Equal(new[] { 27, 37 }, observed);
        Assert.Equal(0, input.HostIndexerCalls);
    }

    private static int ReadOffset(OffsetList view, int index) => view[index];
    private static void WriteOffset(OffsetList view, int index, int value) => view[index] = value;
    private static int ReadNative(List<int> view, int index) => view[index];
    private static int ReadView(IOffsetView view, int index) => view[index];
    private static void WriteView(IOffsetView view, int index, int value) => view[index] = value;
    private static int ReadInt(IIntView view, int index) => view[index];
    private static float ReadFloat(DualStorage view, int index) => view[index];
    private static int ReadInherited(InheritedView view, int index) => view[index];
    private static int ReadChild(IChildView view, int index) => view[index];
    private static int ReadNamed(NamedView view, int index) => view[index];
    private static void WriteNamed(NamedView view, int index, int value) => view[index] = value;
    private static int ReadNamedValue(NamedValueView view, int index) => view[index];

    private sealed class OffsetList(IEnumerable<int> values) : List<int>(values)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<OffsetList, int>(value =>
            {
                value.CaptureCount++;
                return new(value.ToArray(), 1, value.Count, commit: value.Commit, identity: value);
            });
            return true;
        });
        public new int Count => base.Count - 1;
        public new int this[int index] { get => base[index + 1]; set => base[index + 1] = value; }
        public int CaptureCount;
        public int CommitCount;
        public int[]? CommittedValues;
        private void Commit(int[] values) { CommitCount++; CommittedValues = values.ToArray(); }
        public static void Register() => _ = registered.Value;
    }

    private interface IOffsetView
    {
        List<int> Items { get; }
        int Count { get; }
        int this[int index] { get; set; }
        void Committed(int[] values);
        void Captured();
    }

    private sealed class NativeOffsetList(IEnumerable<int> values) : List<int>(values), IOffsetView
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IOffsetView, int>(value =>
            {
                value.Captured();
                return new(value.Items.ToArray(), 1, value.Count, true, value.Committed, value.Items);
            });
            return true;
        });
        List<int> IOffsetView.Items => this;
        int IOffsetView.Count => base.Count - 1;
        int IOffsetView.this[int index] { get => base[index + 1]; set => base[index + 1] = value; }
        public int CaptureCount;
        public int CommitCount;
        public void Committed(int[] values) => CommitCount++;
        public void Captured() => CaptureCount++;
        public static void Register() => _ = registered.Value;
    }

    private interface IIntView { int[] Buffer { get; } int this[int index] { get; set; } }
    private sealed class DualStorage : IIntView
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IIntView, int>(value => new(value.Buffer));
            GpuCollectionAdapters.Register<DualStorage, float>(value => new(value.Buffer));
            return true;
        });
        private readonly int[] ints = [7];
        int[] IIntView.Buffer => ints;
        int IIntView.this[int index] { get => ints[index]; set => ints[index] = value; }
        public float[] Buffer { get; } = [1.5f];
        public float this[int index] { get => Buffer[index]; set => Buffer[index] = value; }
        public static void Register() => _ = registered.Value;
    }

    private class BaseView(int[] buffer)
    {
        public int[] Buffer { get; } = buffer;
        public int Count => Buffer.Length - 1;
        public int HostIndexerCalls;
        public int this[int index]
        {
            get { HostIndexerCalls++; return Buffer[index + 1]; }
            set { HostIndexerCalls++; Buffer[index + 1] = value; }
        }
    }

    private sealed class InheritedView(int[] buffer) : BaseView(buffer)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<InheritedView, int>(value => new(value.Buffer, 1, value.Count));
            return true;
        });
        public static void Register() => _ = registered.Value;
    }

    private interface IParentView { int Count { get; } int this[int index] { get; set; } }
    private interface IChildView : IParentView { int[] Buffer { get; } }
    private sealed class ParentInterfaceView(int[] buffer) : IChildView
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IChildView, int>(value => new(value.Buffer, 1, value.Count));
            return true;
        });
        public int[] Buffer { get; } = buffer;
        public int Count => Buffer.Length - 1;
        public int HostIndexerCalls;
        public int this[int index]
        {
            get { HostIndexerCalls++; return Buffer[index + 1]; }
            set { HostIndexerCalls++; Buffer[index + 1] = value; }
        }
        public static void Register() => _ = registered.Value;
    }

    private sealed class NamedView(int[] buffer)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<NamedView, int>(value => new(value.Buffer, 1, value.Length));
            return true;
        });
        public int[] Buffer { get; } = buffer;
        public int Length => Buffer.Length - 1;
        public int HostIndexerCalls;
        [IndexerName("Elements")]
        public int this[int index]
        {
            get { HostIndexerCalls++; return Buffer[index + 1]; }
            set { HostIndexerCalls++; Buffer[index + 1] = value; }
        }
        public static void Register() => _ = registered.Value;
    }

    private struct NamedValueView(int[] buffer)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<NamedValueView, int>(value => new(value.Buffer, 1, value.Count));
            return true;
        });
        public int[] Buffer = buffer;
        public int Count => Buffer.Length - 1;
        public int HostIndexerCalls;
        [IndexerName("Cells")]
        public int this[int index]
        {
            get { HostIndexerCalls++; return Buffer[index + 1]; }
            set { HostIndexerCalls++; Buffer[index + 1] = value; }
        }
        public static void Register() => _ = registered.Value;
    }
}
