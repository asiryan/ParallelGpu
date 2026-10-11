using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class DeepTransferRegressionTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OneObjectCanExposeIndependentIntAndFloatAdapters(bool floatsFirst)
    {
        MixedStorage.Register();
        var storage = new MixedStorage();
        IIntStorage ints = storage;
        IFloatStorage floats = storage;
        int[] observedInts = [-1];
        float[] observedFloats = [-1];
        if (floatsFirst)
            Gpu.For(0, 1, i =>
            {
                observedFloats[i] = floats[i];
                floats[i] = 2.5f;
                ints[i] += 10;
                observedInts[i] = ints[i];
            });
        else
            Gpu.For(0, 1, i =>
            {
                ints[i] += 10;
                observedInts[i] = ints[i];
                observedFloats[i] = floats[i];
                floats[i] = 2.5f;
            });
        Assert.Equal(new[] { 17 }, storage.Ints);
        Assert.Equal(new[] { 2.5f }, storage.Floats);
        Assert.Equal(new[] { 17 }, observedInts);
        Assert.Equal(new[] { 1.5f }, observedFloats);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RectangularNativeIdentitySharesShapeStorageAndCommitInEitherCaptureOrder(bool nativeFirst)
    {
        MatrixSlice.Register();
        var matrix = (int[,])Array.CreateInstance(typeof(int), [2, 2], [-3, 5]);
        matrix[-3, 5] = 1;
        matrix[-3, 6] = 2;
        matrix[-2, 5] = 3;
        matrix[-2, 6] = 4;
        var view = new MatrixSlice(matrix, 1, 2);
        int[] observed = [-1, -1];
        if (nativeFirst)
            Gpu.For<int>(0, 2, i =>
            {
                int columns = matrix.GetLength(1);
                view[i] += 10;
                int row = (i + 1) / columns + matrix.GetLowerBound(0);
                int column = (i + 1) % columns + matrix.GetLowerBound(1);
                observed[i] = matrix[row, column];
                matrix[row, column] += 2;
            });
        else
            Gpu.For<int>(0, 2, i =>
            {
                view[i] += 10;
                int columns = matrix.GetLength(1);
                int row = (i + 1) / columns + matrix.GetLowerBound(0);
                int column = (i + 1) % columns + matrix.GetLowerBound(1);
                observed[i] = matrix[row, column];
                matrix[row, column] += 2;
            });
        Assert.Equal(new[] { 12, 13 }, observed);
        Assert.Equal(new[] { 1, 14, 15, 4 }, matrix.Cast<int>());
        Assert.Equal(new[] { 1, 14, 15, 4 }, view.CommittedValues);
        Assert.Equal(1, view.CommitCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ListNativeIdentitySharesStorageAndCommitInEitherCaptureOrder(bool nativeFirst)
    {
        ListSlice.Register();
        List<int> list = [1, 2, 3, 4];
        var view = new ListSlice(list, 1, 2);
        int[] observed = [-1, -1];
        if (nativeFirst)
            Gpu.For<int>(0, 2, i =>
            {
                int count = list.Count;
                view[i] += 10;
                observed[i] = list[i + 1] + count - 4;
                list[i + 1] += 2;
            });
        else
            Gpu.For<int>(0, 2, i =>
            {
                view[i] += 10;
                observed[i] = list[i + 1];
                list[i + 1] += 2;
            });
        Assert.Equal(new[] { 12, 13 }, observed);
        Assert.Equal(new[] { 1, 14, 15, 4 }, list);
        Assert.Equal(new[] { 1, 14, 15, 4 }, view.CommittedValues);
        Assert.Equal(1, view.CommitCount);
    }

    [Fact]
    public void ReadOnlyNativeIdentityAdapterSeesNativeWritesAndRejectsWritesWithoutCommitting()
    {
        ListSlice.Register();
        List<int> list = [1, 2, 3, 4];
        var readOnly = new ListSlice(list, 1, 2, true);
        int[] observed = [-1, -1];
        Gpu.For<int>(0, 2, i =>
        {
            int previous = readOnly[i];
            list[i + 1] += previous;
            observed[i] = readOnly[i];
        });
        Assert.Equal(new[] { 1, 4, 6, 4 }, list);
        Assert.Equal(new[] { 4, 6 }, observed);
        Assert.Equal(new[] { 1, 4, 6, 4 }, readOnly.CommittedValues);
        Assert.Equal(1, readOnly.CommitCount);
        Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, i =>
        {
            list[i + 1] = 99;
            observed[i] = 99;
            readOnly[i] = 99;
        }));
        Assert.Equal(new[] { 1, 4, 6, 4 }, list);
        Assert.Equal(new[] { 4, 6 }, observed);
        Assert.Equal(1, readOnly.CommitCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void NativeIdentitiesRejectMismatchedElementTypesAndFullStorageLengths(int kind)
    {
        InvalidNativeSnapshot.Register();
        object identity = kind switch
        {
            0 => new int[2],
            1 => new int[1, 2],
            2 => new float[1],
            3 => new float[1, 1],
            4 => new List<int> { 1, 2 },
            _ => new List<float> { 1.5f }
        };
        var view = new InvalidNativeSnapshot(identity);
        int[] observed = [-1];
        var error = Assert.Throws<InvalidOperationException>(() =>
            Gpu.For<int>(0, 1, i => observed[i] = view[i]));
        Assert.Contains("same element type and full buffer length", error.Message);
        Assert.Equal(new[] { -1 }, observed);
    }

    [Fact]
    public void IndependentTypedViewsCannotClaimTheSameIncompatibleStorageIdentity()
    {
        MixedStorage.Register();
        var storage = new MixedStorage(new object());
        IIntStorage ints = storage;
        IFloatStorage floats = storage;
        float[] observed = [-1];
        var error = Assert.Throws<InvalidOperationException>(() => Gpu.For(0, 1, i =>
        {
            ints[i] = 99;
            observed[i] = floats[i];
        }));
        Assert.Contains("same element type and full buffer length", error.Message);
        Assert.Equal(new[] { 7 }, storage.Ints);
        Assert.Equal(new[] { 1.5f }, storage.Floats);
        Assert.Equal(new[] { -1f }, observed);
    }

    private interface IIntStorage
    {
        int[] Ints { get; }
        object? StorageIdentity { get; }
        int this[int index] { get; set; }
    }

    private interface IFloatStorage
    {
        float[] Floats { get; }
        object? StorageIdentity { get; }
        float this[int index] { get; set; }
    }

    private sealed class MixedStorage(object? storageIdentity = null) : IIntStorage, IFloatStorage
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IIntStorage, int>(value => new(value.Ints, identity: value.StorageIdentity));
            GpuCollectionAdapters.Register<IFloatStorage, float>(value => new(value.Floats, identity: value.StorageIdentity));
            return true;
        });
        public int[] Ints { get; } = [7];
        public float[] Floats { get; } = [1.5f];
        public object? StorageIdentity { get; } = storageIdentity;
        int IIntStorage.this[int index] { get => Ints[index]; set => Ints[index] = value; }
        float IFloatStorage.this[int index] { get => Floats[index]; set => Floats[index] = value; }
        public static void Register() => _ = registered.Value;
    }

    private sealed class MatrixSlice(int[,] matrix, int offset, int count)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<MatrixSlice, int>(value =>
                new(value.matrix.Cast<int>().ToArray(), value.offset, value.count,
                    commit: value.Commit, identity: value.matrix));
            return true;
        });
        private readonly int[,] matrix = matrix;
        private readonly int offset = offset;
        private readonly int count = count;
        public int CommitCount;
        public int[]? CommittedValues;
        public int this[int index]
        {
            get => matrix.GetLength(1) + index;
            set => throw new InvalidOperationException("The adapter indexer must not run on the host.");
        }
        private void Commit(int[] values)
        {
            CommittedValues = values.ToArray();
            CommitCount++;
            for (int i = 0; i < values.Length; i++)
                matrix[i / matrix.GetLength(1) + matrix.GetLowerBound(0),
                    i % matrix.GetLength(1) + matrix.GetLowerBound(1)] = values[i];
        }
        public static void Register() => _ = registered.Value;
    }

    private sealed class ListSlice(List<int> list, int offset, int count, bool readOnly = false)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<ListSlice, int>(value =>
                new(value.list.ToArray(), value.offset, value.count, value.readOnly,
                    commit: value.Commit, identity: value.list));
            return true;
        });
        private readonly List<int> list = list;
        private readonly int offset = offset;
        private readonly int count = count;
        private readonly bool readOnly = readOnly;
        public int CommitCount;
        public int[]? CommittedValues;
        public int this[int index]
        {
            get => list[offset + index];
            set => throw new InvalidOperationException("The adapter indexer must not run on the host.");
        }
        private void Commit(int[] values)
        {
            CommittedValues = values.ToArray();
            CommitCount++;
            for (int i = 0; i < values.Length; i++)
                list[i] = values[i];
        }
        public static void Register() => _ = registered.Value;
    }

    private sealed class InvalidNativeSnapshot(object identity)
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<InvalidNativeSnapshot, int>(value => new([7], identity: value.identity));
            return true;
        });
        private readonly object identity = identity;
        public int this[int index] => 7;
        public static void Register() => _ = registered.Value;
    }
}
