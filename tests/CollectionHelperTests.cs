using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class CollectionHelperTests(GpuFixture fixture)
{
    [Fact]
    public void TransitiveGenericHelpersAcceptArraysListsAndReferences()
    {
        int[] input = [1, 2, 3];
        var result = new List<int> { 0, 0, 0 };
        fixture.Gpu.For(0, input.Length, i => Transform(input, result, i));
        Assert.Equal(new[] { 2, 4, 6 }, result);
        Assert.Equal(new[] { 1, 2, 3 }, input);
    }

    [Fact]
    public void MultidimensionalHelpersKeepArrayShapeAndRefSemantics()
    {
        int[,,] input = new int[2, 3, 4];
        fixture.Gpu.For(0, input.GetLength(0), i => Populate(input, i));
        Assert.Equal(123, input[1, 2, 3]);
    }

    [Fact]
    public void HelpersMayReturnAnExistingCollectionView()
    {
        int[] values = [1, 2, 3];
        fixture.Gpu.For(0, values.Length, i => Identity(values)[i] += 7);
        Assert.Equal(new[] { 8, 9, 10 }, values);
    }

    private static void Transform(int[] input, IList<int> output, int index) =>
        output[index] = DoubleThroughHelper(Read(input, index));

    private static T Read<T>(T[] values, int index) => values[index];
    private static int DoubleThroughHelper(int value) => value * 2;
    private static void Populate(int[,,] values, int i)
    {
        for (int j = 0; j < values.GetLength(1); j++)
            for (int k = 0; k < values.GetLength(2); k++)
                Assign(ref values[i, j, k], i * 100 + j * 10 + k);
    }

    private static void Assign(ref int value, int result) => value = result;
    private static T[] Identity<T>(T[] values) => values;
}

