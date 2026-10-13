using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class TransferIdentityRegressionTests(GpuFixture fixture)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PaddedElementAdaptersCannotShareAnIncompatibleStorageIdentity(bool charsFirst)
    {
        IndexedSlice<bool>.Register();
        IndexedSlice<char>.Register();
        object identity = new();
        var bits = new IndexedSlice<bool>([true], identity);
        var chars = new IndexedSlice<char>(['a'], identity);
        int[] output = [41];
        Exception? error;
        if (charsFirst)
            error = Record.Exception(() => fixture.Gpu.For(0, 1, i =>
            {
                chars[i] = 'z';
                bits[i] = false;
                output[i] = chars[i] + (bits[i] ? 1 : 0);
            }));
        else
            error = Record.Exception(() => fixture.Gpu.For(0, 1, i =>
            {
                bits[i] = false;
                chars[i] = 'z';
                output[i] = (bits[i] ? 1 : 0) + chars[i];
            }));
        Assert.Contains("same element type and full buffer length", Assert.IsType<InvalidOperationException>(error).Message);
        Assert.Equal(new[] { true }, bits.Buffer);
        Assert.Equal(new[] { 'a' }, chars.Buffer);
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(0, bits.CommitCount);
        Assert.Equal(0, chars.CommitCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PaddedListAliasesShareFullStorageAndOneUpdatedCommit(bool nativeFirst)
    {
        BooleanListSlice.Register();
        List<bool> owner = [true, false, true];
        var slice = new BooleanListSlice(owner);
        int[] observed = [-1, -1];
        if (nativeFirst)
            fixture.Gpu.For(0, 2, i =>
            {
                bool before = owner[i + 1];
                slice[i] = !before;
                observed[i] = owner[i + 1] ? 1 : 0;
            });
        else
            fixture.Gpu.For(0, 2, i =>
            {
                slice[i] = !slice[i];
                observed[i] = owner[i + 1] ? 1 : 0;
            });
        Assert.Equal(new[] { true, true, false }, owner);
        Assert.Equal(new[] { 1, 0 }, observed);
        Assert.Equal(new[] { true, true, false }, slice.Committed);
        Assert.Equal(1, slice.CommitCount);
    }

    [Fact]
    public void RegisteredListPassedToNativeHelpersKeepsNativeDeclaration()
    {
        OffsetList.Register();
        var custom = new OffsetList([10, 20, 30]);
        int[] output = [-1, -1, -1, -1];
        fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = custom.Count;
            output[1] = ReadNative(custom, 0);
            output[2] = CountNative(custom);
            output[3] = ReadInterface(custom, 0);
        });
        Assert.Equal(new[] { 2, 10, 3, 10 }, output);
    }

    [Fact]
    public void LocalAndReturnedNativeListAliasesKeepFullStorageAndReferenceIdentity()
    {
        OffsetList.Register();
        var custom = new OffsetList([10, 20, 30]);
        int[] output = new int[6];
        fixture.Gpu.For(0, 1, _ =>
        {
            List<int> local = custom;
            var returned = ReturnNative(custom);
            var readOnly = ReturnReadOnly(custom);
            output[0] = local.Count;
            output[1] = local[0];
            output[2] = returned[2];
            output[3] = readOnly.Count;
            output[4] = readOnly[2];
            output[5] = local == custom && returned == custom && readOnly == custom ? 1 : 0;
        });
        Assert.Equal(new[] { 3, 10, 30, 3, 30, 1 }, output);
    }

    [Fact]
    public void EmptyAdaptedListSliceRetainsItsCompleteNativeStorage()
    {
        OffsetList.Register();
        var custom = new OffsetList([10]);
        int[] output = new int[4];
        fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = custom.Count;
            output[1] = ReadNative(custom, 0);
            output[2] = CountNative(custom);
            output[3] = ReturnReadOnly(custom)[0];
        });
        Assert.Equal(new[] { 0, 10, 1, 10 }, output);
    }

    [Fact]
    public void InheritedListMembersKeepAdaptedSemanticsOnRegisteredReceivers()
    {
        InheritedOffsetList.Register();
        var custom = new InheritedOffsetList([10, 20, 30]);
        int[] output = new int[4];
        fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = custom.Count;
            output[1] = custom[0];
            output[2] = CountNative(custom);
            output[3] = ReadNative(custom, 0);
        });
        Assert.Equal(new[] { 2, 20, 3, 10 }, output);
    }

    [Fact]
    public void NativeListHelperWritesUpdateAdaptedSlicesAndCommitTheFullOwner()
    {
        OffsetList.Register();
        var custom = new OffsetList([10, 20, 30]);
        int[] output = new int[4];
        fixture.Gpu.For(0, 1, _ =>
        {
            WriteNative(custom, 0, 11);
            WriteInterface(custom, 1, 25);
            custom[1] += 7;
            output[0] = ReadNative(custom, 0);
            output[1] = custom[0];
            output[2] = ReadInterface(custom, 2);
            output[3] = CountNative(custom);
        });
        Assert.Equal(new[] { 11, 25, 37, 3 }, output);
        Assert.Equal(new[] { 11, 25, 37 }, (List<int>)custom);
        Assert.Equal(new[] { 11, 25, 37 }, custom.Committed);
        Assert.Equal(1, custom.CommitCount);
    }

    [Fact]
    public void NativeListProjectionIsWritableWhileTheAdapterAliasRemainsReadOnly()
    {
        OffsetList.Register();
        var custom = new OffsetList([10, 20, 30], true);
        int[] output = new int[3];
        fixture.Gpu.For(0, 1, _ =>
        {
            WriteNative(custom, 0, 15);
            WriteInterface(custom, 1, 22);
            output[0] = custom[0];
            output[1] = custom.Count;
            output[2] = ReadNative(custom, 0);
        });
        Assert.Equal(new[] { 22, 2, 15 }, output);
        Assert.Equal(new[] { 15, 22, 30 }, (List<int>)custom);
        Assert.Equal(new[] { 15, 22, 30 }, custom.Committed);
        Assert.Equal(1, custom.CommitCount);

        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            WriteNative(custom, 2, 99);
            output[0] = 99;
            custom[0] = 101;
        }));
        Assert.Equal(new[] { 22, 2, 15 }, output);
        Assert.Equal(new[] { 15, 22, 30 }, (List<int>)custom);
        Assert.Equal(1, custom.CommitCount);
    }

    [Fact]
    public void NativeProjectionBoundsErrorsRetainContextAndCachedBodiesRefreshFullStorage()
    {
        OffsetList.Register();
        var custom = new OffsetList([10, 20, 30]);
        int[] output = [41];
        Action<int> body = _ =>
        {
            WriteNative(custom, 0, 99);
            output[0] = ReadNative(custom, 3);
        };
        Assert.Equal("index", Assert.Throws<ArgumentOutOfRangeException>(() => fixture.Gpu.For(0, 1, body)).ParamName);
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(new[] { 10, 20, 30 }, (List<int>)custom);
        Assert.Equal(0, custom.CommitCount);

        custom.Add(40);
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(new[] { 40 }, output);
        Assert.Equal(new[] { 99, 20, 30, 40 }, (List<int>)custom);
        Assert.Equal(new[] { 99, 20, 30, 40 }, custom.Committed);
        Assert.Equal(1, custom.CommitCount);
    }

    [Fact]
    public void AdapterOverASeparateListOwnerKeepsItsSliceThroughNativeInterfaces()
    {
        IndexedListAdapter.Register();
        var custom = new IndexedListAdapter([10, 20, 30]);
        int[] output = new int[4];
        fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = custom.Count;
            output[1] = ReadInterface(custom, 0);
            output[2] = ReturnReadOnly(custom).Count;
            output[3] = ReturnReadOnly(custom)[1];
        });
        Assert.Equal(new[] { 2, 20, 2, 30 }, output);
    }

    [Fact]
    public void DifferentElementAdapterCannotBePassedToANativeListHelper()
    {
        DifferentElementList.Register();
        var custom = new DifferentElementList([7, 11], [1.5f, 2.5f]);
        Assert.Equal(7, ReadNative(custom, 0));
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = ReadNative(custom, 0);
        }));
        AssertDifferentElementStorageUntouched(custom, output);
    }

    [Fact]
    public void DifferentElementAdapterCannotBeAssignedToANativeListLocal()
    {
        DifferentElementList.Register();
        var custom = new DifferentElementList([7, 11], [1.5f, 2.5f]);
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            List<int> local = custom;
            output[0] = local[0];
        }));
        AssertDifferentElementStorageUntouched(custom, output);
    }

    [Fact]
    public void DifferentElementAdapterCannotEscapeThroughANativeListHelperReturn()
    {
        DifferentElementList.Register();
        var custom = new DifferentElementList([7, 11], [1.5f, 2.5f]);
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = ReturnDifferentElementAsNative(custom)[0];
        }));
        AssertDifferentElementStorageUntouched(custom, output);
    }

    [Fact]
    public void DifferentElementAdapterCannotJoinANativeListReference()
    {
        DifferentElementList.Register();
        var custom = new DifferentElementList([7, 11], [1.5f, 2.5f]);
        List<int> native = [7, 11];
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i =>
        {
            output[0] = 99;
            output[0] = (i == 0 ? custom : native)[0];
        }));
        AssertDifferentElementStorageUntouched(custom, output);
        Assert.Equal(new[] { 7, 11 }, native);
    }

    private static void AssertDifferentElementStorageUntouched(DifferentElementList custom, int[] output)
    {
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(new[] { 7, 11 }, (List<int>)custom);
        Assert.Equal(new[] { 1.5f, 2.5f }, custom.Floats);
        Assert.Equal(0, custom.CaptureCount);
    }

    private static int ReadNative(List<int> values, int index) => values[index];
    private static int CountNative(List<int> values) => values.Count;
    private static int ReadInterface(IReadOnlyList<int> values, int index) => values[index];
    private static void WriteNative(List<int> values, int index, int value) => values[index] = value;
    private static void WriteInterface(IList<int> values, int index, int value) => values[index] = value;
    private static List<int> ReturnNative(List<int> values) => values;
    private static IReadOnlyList<int> ReturnReadOnly(IReadOnlyList<int> values) => values;
    private static List<int> ReturnDifferentElementAsNative(DifferentElementList values) => values;

    private sealed class IndexedSlice<T>(T[] buffer, object identity)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IndexedSlice<T>, T>(value =>
                new(value.Buffer, commit: value.Commit, identity: value.Identity));
            return true;
        });
        public static void Register() => _ = registered.Value;
        public T[] Buffer => buffer;
        public object Identity => identity;
        public int CommitCount;
        private void Commit(T[] values) => CommitCount++;
        public T this[int index] { get => buffer[index]; set => buffer[index] = value; }
    }

    private sealed class BooleanListSlice(List<bool> owner)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<BooleanListSlice, bool>(value =>
                new(value.owner.ToArray(), 1, 2, commit: value.Commit, identity: value.owner));
            return true;
        });
        private readonly List<bool> owner = owner;
        public static void Register() => _ = registered.Value;
        public int CommitCount;
        public bool[]? Committed;
        private void Commit(bool[] values) { CommitCount++; Committed = values.ToArray(); }
        public bool this[int index] { get => owner[index + 1]; set => owner[index + 1] = value; }
    }

    private sealed class OffsetList(IEnumerable<int> values, bool readOnly = false) : List<int>(values)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<OffsetList, int>(value =>
                new(value.ToArray(), 1, value.Count, value.readOnly, value.Commit, value));
            return true;
        });
        private readonly bool readOnly = readOnly;
        public static void Register() => _ = registered.Value;
        public int CommitCount;
        public int[]? Committed;
        private void Commit(int[] values) { CommitCount++; Committed = values.ToArray(); }
        public new int Count => base.Count - 1;
        public new int this[int index] { get => base[index + 1]; set => base[index + 1] = value; }
    }

    private sealed class IndexedListAdapter(List<int> owner) : IReadOnlyList<int>
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IndexedListAdapter, int>(value => new(value.owner.ToArray(), 1, 2, identity: value.owner));
            return true;
        });
        private readonly List<int> owner = owner;
        public static void Register() => _ = registered.Value;
        public int Count => 2;
        public int this[int index] => owner[index + 1];
        public IEnumerator<int> GetEnumerator() => owner.Skip(1).GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private sealed class InheritedOffsetList(IEnumerable<int> values) : List<int>(values)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<InheritedOffsetList, int>(value => new(value.ToArray(), 1, 2, identity: value));
            return true;
        });
        public static void Register() => _ = registered.Value;
    }

    private sealed class DifferentElementList(IEnumerable<int> values, float[] floats) : List<int>(values)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<DifferentElementList, float>(value =>
            {
                value.CaptureCount++;
                return new(value.Floats);
            });
            return true;
        });
        public static void Register() => _ = registered.Value;
        public float[] Floats { get; } = floats;
        public int CaptureCount;
        public new float this[int index] { get => Floats[index]; set => Floats[index] = value; }
    }
}
