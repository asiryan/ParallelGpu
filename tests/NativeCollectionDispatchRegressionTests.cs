using System.Collections;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class NativeCollectionDispatchRegressionTests(GpuFixture fixture)
{
    static NativeCollectionDispatchRegressionTests()
    {
        Register<SplitCustomIndexer>();
        Register<SplitCustomCount>();
        Register<CompatibleCustom>();
        Register<NotificationView>();
        GpuCollectionAdapters.Register<IBaseCountView, int>(owner => new(owner.Storage, 0, 1));
        GpuCollectionAdapters.Register<IDerivedCountView, int>(owner => new(owner.Storage, 0, 2));
    }

    private static void Register<T>() where T : CustomView =>
        GpuCollectionAdapters.Register<T, int>(owner =>
        {
            owner.CaptureCount++;
            return new(owner.Storage);
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReimplementedNativeIndexerRejectsInterfaceAndListDeclarationsBeforeCallbacks(bool listDeclaration)
    {
        var owner = new SplitNativeIndexer();
        var notification = new NotificationView();
        int[] output = [41];
        Exception? error;
        if (listDeclaration)
        {
            List<int> values = owner;
            error = Record.Exception(() => fixture.Gpu.For(0, 1, i =>
            {
                output[i] = notification[i];
                WriteNative(values, i, 123);
            }));
        }
        else
        {
            IList<int> values = owner;
            error = Record.Exception(() => fixture.Gpu.For(0, 1, i =>
            {
                output[i] = notification[i];
                values[i] += 20;
            }));
        }
        Assert.Contains("reimplements", Assert.IsType<NotSupportedException>(error).Message);
        Assert.Equal(new[] { 7 }, (List<int>)owner);
        Assert.Equal(new[] { 100 }, owner.OtherStorage);
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(0, notification.CaptureCount);
        fixture.Gpu.For(0, 1, i => output[i] = notification[i] + 1);
        Assert.Equal(new[] { 8 }, output);
    }

    [Fact]
    public void ReimplementedReadOnlyNativeIndexerIsRejected()
    {
        var owner = new SplitNativeReadOnlyIndexer();
        IReadOnlyList<int> values = owner;
        int[] output = [41];
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i => output[i] = values[i]));
        Assert.Equal(100, values[0]);
        Assert.Equal(new[] { 7 }, (List<int>)owner);
        Assert.Equal(new[] { 41 }, output);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReimplementedNativeCountIsRejected(bool readOnly)
    {
        int[] output = [41];
        if (readOnly)
        {
            IReadOnlyList<int> values = new SplitNativeReadOnlyCount();
            Assert.Equal(100, values.Count);
            Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i => output[i] = values.Count));
        }
        else
        {
            IList<int> values = new SplitNativeCount();
            Assert.Equal(100, values.Count);
            Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i => output[i] = values.Count));
        }
        Assert.Equal(new[] { 41 }, output);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeAdapterCannotReplaceAnExplicitInterfaceIndexer(bool readOnly)
    {
        var owner = new SplitCustomIndexer();
        int[] output = [41];
        if (readOnly)
        {
            IReadOnlyList<int> values = owner;
            Assert.Equal(100, values[0]);
            Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i =>
            {
                output[i] = owner[i];
                output[i] = values[i];
            }));
        }
        else
        {
            IList<int> values = owner;
            Assert.Equal(100, values[0]);
            Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i =>
            {
                output[i] = owner[i];
                values[i] += 20;
            }));
        }
        Assert.Equal(new[] { 7 }, owner.Storage);
        Assert.Equal(new[] { 100 }, owner.OtherStorage);
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(0, owner.CaptureCount);
    }

    [Fact]
    public void RuntimeAdapterCannotReplaceAnExplicitInterfaceCount()
    {
        var owner = new SplitCustomCount();
        IList<int> values = owner;
        int[] output = [41];
        Assert.Equal(100, values.Count);
        Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i => output[i] = values.Count));
        Assert.Equal(new[] { 7 }, owner.Storage);
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(0, owner.CaptureCount);
    }

    [Fact]
    public void RuntimeAdapterKeepsCompatiblePublicInterfaceAccessorsAndCachedFreshStorage()
    {
        var first = new CompatibleCustom();
        IList<int> values = first;
        IReadOnlyList<int> readOnly = first;
        int[] output = [41];
        Action<int> body = i =>
        {
            values[i] += 20;
            output[i] = ReadReadOnly(readOnly, i) + values.Count;
        };
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(new[] { 27 }, first.Storage);
        Assert.Equal(new[] { 28 }, output);

        var second = new CompatibleCustom();
        second.Storage[0] = 50;
        values = second;
        readOnly = second;
        fixture.Gpu.For(0, 1, body);
        Assert.Equal(new[] { 27 }, first.Storage);
        Assert.Equal(new[] { 70 }, second.Storage);
        Assert.Equal(new[] { 71 }, output);
    }

    [Fact]
    public void InheritedNativeInterfacesIgnoreNewPublicIndexersAndCount()
    {
        var owner = new NewPublicNativeMembers();
        IList<int> values = owner;
        IReadOnlyList<int> readOnly = owner;
        List<int> native = owner;
        int[] output = [-1, -1, -1, -1];
        fixture.Gpu.For(0, 1, i =>
        {
            values[i] += 20;
            output[0] = values[i];
            output[1] = ReadReadOnly(readOnly, i);
            output[2] = values.Count;
            output[3] = ReadNative(native, i);
        });
        Assert.Equal(new[] { 27, 27, 1, 27 }, output);
        Assert.Equal(new[] { 27 }, (List<int>)owner);
        Assert.Equal(100, owner[0]);
        Assert.Equal(100, owner.Count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HiddenRegisteredCountCannotReuseTheDerivedSnapshotThroughAnUpcast(bool helper)
    {
        var owner = new SplitRegisteredCount();
        IDerivedCountView derived = owner;
        int[] output = [41];
        Assert.Equal(2, derived.Count);
        Assert.Equal(1, ((IBaseCountView)derived).Count);
        if (helper)
            Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i =>
            {
                output[i] = 99;
                output[i] = ReadBaseCount(derived);
            }));
        else
            Assert.Throws<NotSupportedException>(() => fixture.Gpu.For(0, 1, i =>
            {
                output[i] = 99;
                output[i] = ((IBaseCountView)derived).Count;
            }));
        Assert.Equal(new[] { 41 }, output);
        Assert.Equal(new[] { 7, 100 }, owner.Storage);

        IBaseCountView declaredBase = owner;
        fixture.Gpu.For(0, 1, i => output[i] = derived.Count * 10 + declaredBase.Count);
        Assert.Equal(new[] { 21 }, output);
    }

    private static int ReadNative(List<int> values, int index) => values[index];
    private static void WriteNative(List<int> values, int index, int value) => values[index] = value;
    private static int ReadReadOnly(IReadOnlyList<int> values, int index) => values[index];
    private static int ReadBaseCount(IBaseCountView values) => values.Count;

    private interface IBaseCountView
    {
        int[] Storage { get; }
        int Count { get; }
        int this[int index] { get; }
    }

    private interface IDerivedCountView : IBaseCountView
    {
        new int Count { get; }
    }

    private sealed class SplitRegisteredCount : IDerivedCountView
    {
        public int[] Storage { get; } = [7, 100];
        public int this[int index] => Storage[index];
        int IBaseCountView.Count => 1;
        int IDerivedCountView.Count => 2;
    }

    private sealed class SplitNativeIndexer() : List<int>([7]), IList<int>
    {
        public int[] OtherStorage { get; } = [100];
        int IList<int>.this[int index] { get => OtherStorage[index]; set => OtherStorage[index] = value; }
    }

    private sealed class SplitNativeReadOnlyIndexer() : List<int>([7]), IReadOnlyList<int>
    {
        int IReadOnlyList<int>.this[int index] => 100;
    }

    private sealed class SplitNativeCount() : List<int>([7]), ICollection<int>
    {
        int ICollection<int>.Count => 100;
    }

    private sealed class SplitNativeReadOnlyCount() : List<int>([7]), IReadOnlyCollection<int>
    {
        int IReadOnlyCollection<int>.Count => 100;
    }

    private sealed class NewPublicNativeMembers() : List<int>([7])
    {
        public new int this[int index] => 100;
        public new int Count => 100;
    }

    private sealed class SplitCustomIndexer : CustomView, IList<int>, IReadOnlyList<int>
    {
        public int[] OtherStorage { get; } = [100];
        int IList<int>.this[int index] { get => OtherStorage[index]; set => OtherStorage[index] = value; }
        int IReadOnlyList<int>.this[int index] => OtherStorage[index];
    }

    private sealed class SplitCustomCount : CustomView, ICollection<int>
    {
        int ICollection<int>.Count => 100;
    }

    private sealed class CompatibleCustom : CustomView;
    private sealed class NotificationView : CustomView;

    private abstract class CustomView : IList<int>, IReadOnlyList<int>
    {
        public int[] Storage { get; } = [7];
        public int CaptureCount;
        public int Count => Storage.Length;
        public bool IsReadOnly => false;
        public int this[int index] { get => Storage[index]; set => Storage[index] = value; }
        public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)Storage).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        public void Add(int item) => throw new NotSupportedException();
        public void Clear() => throw new NotSupportedException();
        public bool Contains(int item) => Storage.Contains(item);
        public void CopyTo(int[] array, int index) => Storage.CopyTo(array, index);
        public int IndexOf(int item) => Array.IndexOf(Storage, item);
        public void Insert(int index, int item) => throw new NotSupportedException();
        public bool Remove(int item) => throw new NotSupportedException();
        public void RemoveAt(int index) => throw new NotSupportedException();
    }
}
