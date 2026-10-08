using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ArrayShapeTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Fact]
    public void ThreeDimensionalArraysSupportReadWriteReferencesAndShapeQueries()
    {
        var values = new int[3, 2, 4];
        var output = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
            values[i / 8, i / 4 % 2, i % 4] = i * 3;

        Gpu.For<int>(0, values.GetLength(0), i =>
        {
            for (int j = 0; j < values.GetLength(1); j++)
                for (int k = 0; k < values.GetLength(2); k++)
                {
                    Increment(ref values[i, j, k]);
                    int flat = (i * values.GetLength(1) + j) * values.GetLength(2) + k;
                    output[flat] = values[i, j, k] + values.Rank + (int)values.GetLongLength(2);
                }
        });

        Assert.Equal(Enumerable.Range(0, values.Length).Select(i => i * 3 + 1), values.Cast<int>());
        Assert.Equal(Enumerable.Range(0, values.Length).Select(i => i * 3 + 8), output);
    }

    [Fact]
    public void FourDimensionalArraysRetainEveryNonzeroLowerBound()
    {
        var values = (int[,,,])Array.CreateInstance(typeof(int), [2, 3, 2, 2], [-3, 4, 7, -8]);
        var output = new int[values.Length];
        int next = 0;
        for (int i = -3; i <= -2; i++)
            for (int j = 4; j <= 6; j++)
                for (int k = 7; k <= 8; k++)
                    for (int l = -8; l <= -7; l++)
                        values[i, j, k, l] = next++;

        Gpu.For<int>(0, values.GetLength(0), row =>
        {
            int i = row + values.GetLowerBound(0);
            for (int j = values.GetLowerBound(1); j <= values.GetUpperBound(1); j++)
                for (int k = values.GetLowerBound(2); k <= values.GetUpperBound(2); k++)
                    for (int l = values.GetLowerBound(3); l <= values.GetUpperBound(3); l++)
                    {
                        values[i, j, k, l] += 17;
                        int flat = ((row * 3 + j - 4) * 2 + k - 7) * 2 + l + 8;
                        output[flat] = values[i, j, k, l] + values.GetLowerBound(0) + values.GetUpperBound(3);
                    }
        });

        Assert.Equal(Enumerable.Range(0, values.Length).Select(i => i + 17), values.Cast<int>());
        Assert.Equal(Enumerable.Range(0, values.Length).Select(i => i + 7), output);
        Assert.Equal(-3, values.GetLowerBound(0));
        Assert.Equal(-7, values.GetUpperBound(3));
    }

    [Fact]
    public void PrivateStructuresFlattenWithoutBoxingTheirFields()
    {
        var values = new Pair[2, 2, 3];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 3; k++)
                    values[i, j, k] = new Pair { Number = (i * 2 + j) * 3 + k, Symbol = '\uABCD' };

        Gpu.For<Pair>(0, 2, i =>
        {
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 3; k++)
                {
                    values[i, j, k].Number += 5;
                    values[i, j, k].Symbol = (char)(values[i, j, k].Symbol + 1);
                }
        });

        Assert.Equal(Enumerable.Range(0, values.Length).Select(i => i + 5), values.Cast<Pair>().Select(p => p.Number));
        Assert.All(values.Cast<Pair>(), p => Assert.Equal('\uABCE', p.Symbol));
    }

    [Fact]
    public void EmptyHighRankArraysRetainTheirDimensions()
    {
        var values = new int[2, 0, 3, 4, 5];
        var output = new int[7];
        Gpu.For<int>(0, 1, _ =>
        {
            output[0] = values.Length;
            output[1] = values.Rank;
            for (int d = 0; d < values.Rank; d++)
                output[d + 2] = values.GetLength(d);
        });
        Assert.Equal(new[] { 0, 5, 2, 0, 3, 4, 5 }, output);
    }

    [Fact]
    public void DeepJaggedArraysPreserveContainerAndLeafAliasesNullsAndEmptyNodes()
    {
        int[] leaf = [2, 4, 6];
        int[][] branch = [leaf, null!, [], leaf];
        int[][][] middle = [branch, branch, null!, []];
        int[][][][] tree = [middle, middle];
        int[] output = new int[8];

        Gpu.For<int>(0, 1, _ =>
        {
            var first = tree[0];
            var second = tree[1];
            output[0] = first == second ? 1 : 0;
            output[1] = first[0] == first[1] ? 1 : 0;
            output[2] = first[2] == null ? 1 : 0;
            output[3] = first[0][1] == null ? 1 : 0;
            output[4] = first[3].Length;
            output[5] = first[0][2].Length;
            output[6] = first[0][0] == leaf ? 1 : 0;
            output[7] = first[0][2] == null ? 1 : 0;
            for (int i = 0; i < leaf.Length; i++)
            {
                first[0][0][i] += 3;
                second[1][3][i] *= 2;
                leaf[i] += 1;
            }
        });

        Assert.Equal(new[] { 1, 1, 1, 1, 0, 0, 1, 0 }, output);
        Assert.Equal(new[] { 11, 15, 19 }, leaf);
        Assert.Same(tree[0], tree[1]);
        Assert.Same(tree[0][0], tree[0][1]);
        Assert.Null(tree[0][2]);
        Assert.Null(tree[0][0][1]);
    }

    [Fact]
    public void RectangularContainersCanContainRectangularLeaves()
    {
        int[,] shared = { { 2, 3 }, { 5, 7 } };
        var container = new int[2, 2][,];
        container[0, 0] = shared;
        container[1, 1] = shared;
        container[0, 1] = new int[0, 3];
        container[1, 0] = null!;
        int[] output = new int[3];

        Gpu.For<int>(0, 1, _ =>
        {
            var leaf = container[0, 0];
            output[0] = container[1, 0] == null ? 1 : 0;
            output[1] = container[0, 1].GetLength(1);
            output[2] = leaf == shared && leaf == container[1, 1] ? 1 : 0;
            for (int i = 0; i < leaf.GetLength(0); i++)
                for (int j = 0; j < leaf.GetLength(1); j++)
                {
                    leaf[i, j] += 1;
                    container[1, 1][i, j] *= 3;
                }
        });

        Assert.Equal(new[] { 1, 3, 1 }, output);
        Assert.Equal(new[] { 9, 12, 18, 24 }, shared.Cast<int>());
        Assert.Same(container[0, 0], container[1, 1]);
    }

    [Fact]
    public void InvalidDimensionIndexDoesNotAliasAValidFlatElementOrCopyBackWrites()
    {
        var values = new int[2, 2, 2];
        int[] output = [41];
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            values[0, 2, 0] = 7;
        }));
        Assert.Equal(41, output[0]);
        Assert.All(values.Cast<int>(), value => Assert.Equal(0, value));
    }

    [Fact]
    public void NullNestedDereferenceFailsBeforeCopyBack()
    {
        int[][][] values = [[null!]];
        int[] output = [41];
        Assert.Throws<NullReferenceException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = values[0][0].Length;
        }));
        Assert.Equal(41, output[0]);
        Assert.Null(values[0][0]);
    }

    [Fact]
    public void InvalidReferenceToEmptyLeafUsesSafeScratchAndDoesNotCopyBack()
    {
        int[][] values = [[]];
        int[] output = [41];
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            Increment(ref values[0][0]);
        }));
        Assert.Equal(41, output[0]);
        Assert.Empty(values[0]);
    }

    [Fact]
    public void CachedBodiesReloadShapeLowerBoundsAndNestedAliases()
    {
        var values = (int[,,])Array.CreateInstance(typeof(int), [1, 2, 2], [3, -1, 8]);
        int[] output = new int[2];
        int[] shared = [2];
        int[][][] nodes = [[shared], [shared]];
        Action<int> body = _ =>
        {
            output[0] = values.GetLength(1) + values.GetLowerBound(0) + values.GetUpperBound(2);
            output[1] = nodes[0][0] == nodes[1][0] ? 1 : 0;
            for (int j = values.GetLowerBound(1); j <= values.GetUpperBound(1); j++)
                for (int k = values.GetLowerBound(2); k <= values.GetUpperBound(2); k++)
                    values[values.GetLowerBound(0), j, k] = j * 100 + k;
        };

        Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { 14, 1 }, output);
        Assert.Equal(new[] { -92, -91, 8, 9 }, values.Cast<int>());

        values = (int[,,])Array.CreateInstance(typeof(int), [1, 1, 3], [-5, 7, -2]);
        nodes = [[[3]], [[4]]];
        Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { -4, 0 }, output);
        Assert.Equal(new[] { 698, 699, 700 }, values.Cast<int>());
    }

    [Fact]
    public void ArrayHelpersReceiveNestedAndMultidimensionalViews()
    {
        int[,,] values = { { { 2, 3 } }, { { 5, 7 } } };
        int[][][] offsets = [[[11]], [[13]]];
        int[] output = new int[2];
        Gpu.For<int>(0, output.Length, row => output[row] = AddRow(values, offsets, row));
        Assert.Equal(new[] { 16, 25 }, output);
    }

    private static int AddRow(int[,,] values, int[][][] offsets, int row)
    {
        int sum = offsets[row][0][0];
        for (int i = 0; i < values.GetLength(2); i++)
            sum += values[row, 0, i];
        return sum;
    }

    [Fact]
    public void EmptyArraysWithLargeTheoreticalStridesAndAllNullTreesKeepTheirShapes()
    {
        var values = new int[0, 50_000, 50_000];
        int[][][][] nodes = [null!, []];
        int[] output = new int[5];
        Gpu.For<int>(0, 1, _ =>
        {
            output[0] = values.Length;
            output[1] = values.GetLength(1);
            output[2] = values.GetLength(2);
            output[3] = nodes[0] == null ? 1 : 0;
            output[4] = nodes[1].Length;
        });
        Assert.Equal(new[] { 0, 50_000, 50_000, 1, 0 }, output);
    }

    [Fact]
    public void LocalAndHelperReturnedNullViewsFailBeforeCopyBack()
    {
        int[] output = [41];
        Assert.Throws<NullReferenceException>(() => Gpu.For<int>(0, 1, _ =>
        {
            int[] local = null!;
            output[0] = 99;
            output[0] = local.Length;
        }));
        Assert.Equal(41, output[0]);
        Assert.Throws<NullReferenceException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = NullArray().Length;
        }));
        Assert.Equal(41, output[0]);
    }

    [Fact]
    public void CapturedNullViewsCanBeGuardedAndDereferencesFailBeforeCopyBack()
    {
        int[]? missing = null;
        int[] output = [41];
        Action<int> guard = _ => output[0] = missing == null ? 7 : missing.Length;
        Gpu.For<int>(0, 1, guard);
        Assert.Equal(7, output[0]);
        missing = [1, 2, 3];
        Gpu.For<int>(0, 1, guard);
        Assert.Equal(3, output[0]);
        missing = null;
        Assert.Throws<NullReferenceException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            output[0] = missing!.Length;
        }));
        Assert.Equal(3, output[0]);
    }

    [Fact]
    public void InvalidReferencesInsideNestedHelpersUseSafeScratch()
    {
        int[][] rows = [[]];
        int[,,] tensor = new int[0, 2, 3];
        int[] output = [41];
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            IncrementNested(rows);
        }));
        Assert.Equal(41, output[0]);
        Assert.Throws<IndexOutOfRangeException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            IncrementTensor(tensor);
        }));
        Assert.Equal(41, output[0]);
    }

    [Fact]
    public void InvalidReferencesToLocalLargeElementsReserveEnoughScratch()
    {
        int[] output = [41];
        Assert.Throws<NullReferenceException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            var local = NullPayloadArray();
            FillPayload(ref local[0]);
        }));
        Assert.Equal(41, output[0]);
        Assert.Throws<NullReferenceException>(() => Gpu.For<int>(0, 1, _ =>
        {
            output[0] = 99;
            var local = NullPayloadTensor();
            FillPayload(ref local[0, 0, 0]);
        }));
        Assert.Equal(41, output[0]);
    }

    private static void IncrementNested(int[][] rows) => IncrementFirst(rows[0]);
    private static void IncrementFirst(int[] row) => Increment(ref row[0]);
    private static void IncrementTensor(int[,,] tensor) => Increment(ref tensor[0, 0, 0]);
    private static LargePayload[] NullPayloadArray() => null!;
    private static LargePayload[,,] NullPayloadTensor() => null!;
    private static void FillPayload(ref LargePayload value)
    {
        value.A = long.MaxValue;
        value.B = long.MinValue;
        value.C = 1234567890123;
        value.D = -1234567890123;
    }

    private struct LargePayload
    {
        public long A, B, C, D;
    }

    private static int[] NullArray() => null!;

    private static void Increment(ref int value) => value++;

    private struct Pair
    {
        public int Number;
        public char Symbol;
    }
}
