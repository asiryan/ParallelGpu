using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class LambdaLoopTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void LargeTransfersPreserveAliasesUntouchedElementsAndFreshHostValues()
    {
        // The three distinct leaves, including the empty row, average over 1 MiB.
        int[] input = Enumerable.Range(0, 500_001).ToArray();
        int[] output = Enumerable.Repeat(-1, 500_003).ToArray();
        int[] alias = output;
        int[][] rows = [input, output, alias, Array.Empty<int>()];
        Action<int> body = i =>
        {
            rows[1][i] = input[i] * 2;
            alias[i] += rows[3].Length + 1;
        };

        Gpu.For<int>(7, 258, body);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(i >= 7 && i < 258 ? i * 2 + 1 : -1, output[i]);
        for (int i = 0; i < input.Length; i++)
            Assert.Equal(i, input[i]);

        input[17] = -42;
        output[17] = 900;
        Gpu.For<int>(7, 258, body);
        Assert.Equal(-83, output[17]);
        Assert.Equal(-42, input[17]);
        Assert.Equal(-1, output[6]);
        Assert.Equal(-1, output[258]);
        Assert.Same(rows[1], rows[2]);
    }

    [Fact]
    public void FloatArraysSupportLocalsBranchesAndInnerLoops()
    {
        float[] left = [1, -2, 3];
        float[] right = [2, 3, 4];
        float[] result = new float[left.Length];
        float factor = 2;

        Gpu.For<float>(0, result.Length, i =>
        {
            float sum = 0;
            for (int k = 0; k < 3; k++)
                sum += left[i] * right[i];
            result[i] = sum < 0 ? -sum : sum * factor;
            left[i] += factor;
        });

        Assert.Equal(new float[] { 12, 18, 72 }, result);
        Assert.Equal(new float[] { 3, 0, 5 }, left);
        Assert.Equal(new float[] { 2, 3, 4 }, right);
    }

    [Fact]
    public void ListsSupportIndexingCountAndInterfaceReferences()
    {
        IList<int> input = new List<int> { 3, 5, 7 };
        IReadOnlyList<int> offsets = new[] { 1, 2, 3 };
        var result = new List<int> { 0, 0, 0 };

        using var inputEnumerator = input.GetEnumerator();
        Gpu.For<int>(0, result.Count, i => result[i] = input[i] + offsets[i] + input.Count);

        Assert.Equal(new[] { 7, 10, 13 }, result);
        Assert.Equal(new[] { 3, 5, 7 }, input);
        Assert.True(inputEnumerator.MoveNext());
        Assert.Equal(3, inputEnumerator.Current);
    }

    [Fact]
    public void CachedBodyReadsFreshScalarsAndCollections()
    {
        int[] output = new int[3];
        int factor = 3;
        Action<int> body = i => output[i] = i * factor;

        Gpu.For<int>(0, output.Length, body);
        Assert.Equal(new[] { 0, 3, 6 }, output);

        output = new int[4];
        factor = 7;
        Gpu.For<int>(0, output.Length, body);
        Assert.Equal(new[] { 0, 7, 14, 21 }, output);
    }

    [Fact]
    public void ClosedGenericStaticFieldsKeepSeparateCapturesAndReadFreshValues()
    {
        GenericCaptureState<int>.Scalar = 2;
        GenericCaptureState<float>.Scalar = 7;
        GenericCaptureState<int>.Values = [3, 4];
        GenericCaptureState<float>.Values = [10, 20];
        int[] scalarResult = new int[2];
        int[] arrayResult = new int[2];

        Action<int> body = i =>
        {
            scalarResult[i] = GenericCaptureState<int>.Scalar + GenericCaptureState<float>.Scalar + i;
            arrayResult[i] = GenericCaptureState<int>.Values[i] + GenericCaptureState<float>.Values[i];
            GenericCaptureState<int>.Values[i] += GenericCaptureState<int>.Scalar;
            GenericCaptureState<float>.Values[i] += GenericCaptureState<float>.Scalar;
        };

        Gpu.For<int>(0, scalarResult.Length, body);
        Assert.Equal(new[] { 9, 10 }, scalarResult);
        Assert.Equal(new[] { 13, 24 }, arrayResult);
        Assert.Equal(new[] { 5, 6 }, GenericCaptureState<int>.Values);
        Assert.Equal(new[] { 17, 27 }, GenericCaptureState<float>.Values);

        GenericCaptureState<int>.Scalar = 3;
        GenericCaptureState<float>.Scalar = 11;
        GenericCaptureState<int>.Values = [1, 2];
        GenericCaptureState<float>.Values = [20, 30];

        Gpu.For<int>(0, scalarResult.Length, body);
        Assert.Equal(new[] { 14, 15 }, scalarResult);
        Assert.Equal(new[] { 21, 32 }, arrayResult);
        Assert.Equal(new[] { 4, 5 }, GenericCaptureState<int>.Values);
        Assert.Equal(new[] { 31, 41 }, GenericCaptureState<float>.Values);
    }

    [Fact]
    public void CachedBodyUsesTheCurrentClosureInstance()
    {
        int[] first = new int[3];
        int[] second = new int[3];

        Gpu.For<int>(0, 3, MakeBody(first, 3));
        Gpu.For<int>(0, 3, MakeBody(second, 5));

        Assert.Equal(new[] { 0, 3, 6 }, first);
        Assert.Equal(new[] { 0, 5, 10 }, second);
    }

    private static Action<int> MakeBody(int[] output, int multiplier) =>
        i => output[i] = i * multiplier;

    [Fact]
    public void JaggedArraysSupportRaggedAndEmptyRows()
    {
        float[][] input = [[1, 2], [], [4]];
        float[][] output = [[0, 0], [], [0]];

        Gpu.For<float>(0, input.Length, row =>
        {
            var values = input[row];
            for (int column = 0; column < values.Length; column++)
                output[row][column] = values[column] * 2 + row;
        });

        Assert.Equal(new float[] { 2, 4 }, output[0]);
        Assert.Empty(output[1]);
        Assert.Equal(new float[] { 10 }, output[2]);

        float[][] emptyRows = [[], []];
        int[] lengths = [-1, -1];
        Gpu.For(0, lengths.Length, row => lengths[row] = emptyRows[row].Length);
        Assert.Equal(new[] { 0, 0 }, lengths);
    }

    [Fact]
    public void JaggedRowsAndCapturedArraysPreserveAliases()
    {
        int[] shared = [1, 2, 3];
        int[][] rows = [shared, shared];

        Gpu.For<int>(0, shared.Length, i =>
        {
            rows[0][i] += 2;
            rows[1][i] *= 3;
            shared[i] *= 2;
        });

        Assert.Equal(new[] { 18, 24, 30 }, shared);
        Assert.Same(shared, rows[0]);
        Assert.Same(shared, rows[1]);
    }

    [Fact]
    public void ListAliasesShareTheSameDeviceData()
    {
        var values = new List<float> { 1, 2, 3 };
        IList<float> alias = values;

        Gpu.For<float>(0, values.Count, i =>
        {
            values[i] += 3;
            alias[i] *= 2;
        });

        Assert.Equal(new float[] { 8, 10, 12 }, values);
    }

    [Fact]
    public void RectangularArraysSupportDimensionsAndElementReferences()
    {
        int[,] values = { { 1, 2, 3 }, { 4, 5, 6 } };
        int[,] result = new int[2, 3];

        Gpu.For<int>(0, values.GetLength(0), row =>
        {
            for (int column = 0; column < values.GetLength(1); column++)
            {
                Increment(ref values[row, column]);
                result[row, column] = values[row, column] * 2 + values.Length;
            }
        });

        Assert.Equal(new[] { 2, 3, 4, 5, 6, 7 }, values.Cast<int>());
        Assert.Equal(new[] { 10, 12, 14, 16, 18, 20 }, result.Cast<int>());
    }

    private static void Increment(ref int value) => value++;

    [Fact]
    public void MixedCollectionsRequireTheNonGenericOverload()
    {
        int[] input = [1, 2, 3];
        float[] output = [-1, -1, -1];

        Assert.Throws<ArgumentException>(() =>
            Gpu.For<float>(0, output.Length, i => output[i] = input[i]));
        Assert.Equal(new float[] { -1, -1, -1 }, output);

        Gpu.For(0, output.Length, i => output[i] = input[i] * 1.5f);
        Assert.Equal(new float[] { 1.5f, 3, 4.5f }, output);
    }

    [Fact]
    public void SupportedNumericTypesRoundTrip()
    {
        RoundTrip<uint>(uint.MaxValue);
        RoundTrip<long>(long.MinValue);
        RoundTrip<ulong>(ulong.MaxValue);
        RoundTrip<double>(-123.125);
        RoundTrip<short>(short.MinValue);
        RoundTrip<ushort>(ushort.MaxValue);
        RoundTrip<byte>(byte.MaxValue);
        RoundTrip<sbyte>(sbyte.MinValue);
    }

    private void RoundTrip<T>(T value) where T : unmanaged
    {
        T[] input = [value, value];
        T[] output = new T[2];
        Gpu.For<T>(0, output.Length, i => output[i] = input[i]);
        Assert.Equal(input, output);
    }

    [Fact]
    public void RangeLimitsPreserveTheLoopContract()
    {
        int[] values = Enumerable.Repeat(-1, 260).ToArray();
        Gpu.For<int>(1, 258, i => values[i] = i);
        Assert.Equal(-1, values[0]);
        Assert.Equal(Enumerable.Range(1, 257), values.Skip(1).Take(257));
        Assert.Equal(-1, values[258]);

        int[] negative = new int[3];
        Gpu.For<int>(-2, 1, i => negative[i + 2] = i);
        Assert.Equal(new[] { -2, -1, 0 }, negative);

        Gpu.For<int>(5, 5, _ => throw new InvalidOperationException());
        Gpu.For<int>(5, -5, _ => throw new InvalidOperationException());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Gpu.For<int>(int.MinValue, int.MaxValue, _ => { }));
    }

    [Fact]
    public void UnsupportedBodiesDoNotExecuteOrPreventFollowingLoops()
    {
        var values = new List<int> { 1 };
        Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, i => values.Add(i)));
        Assert.Equal(new[] { 1 }, values);

        Assert.Throws<NotSupportedException>(() => Gpu.For(0, 1, i => Console.WriteLine(i)));
        int[] output = new int[3];
        Gpu.For<int>(0, output.Length, i => output[i] = i + 2);
        Assert.Equal(new[] { 2, 3, 4 }, output);
    }

    [Fact]
    public void StaticFacadeRunsATypedBodyOnTheSelectedBackend()
    {
        float[] output = new float[3];

        ParallelGpu.GpuParallel.For<float>(0, output.Length, new() { Backend = Gpu.Backend },
            i => output[i] = i * 0.5f);

        Assert.Equal(new float[] { 0, 0.5f, 1 }, output);
    }
}
