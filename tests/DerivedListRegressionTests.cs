using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class DerivedListRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void CachedInterfaceCapturesAcceptNonGenericAndGenericListSubclasses()
    {
        IList<int> values = new DerivedList { 7, 11 };
        IReadOnlyList<int> readOnly = (IReadOnlyList<int>)values;
        int[] output = [-1, -1];
        Action<int> body = i =>
        {
            values[i] += 10;
            output[i] = readOnly[i] + values.Count;
        };

        fixture.Gpu.For<int>(0, 2, body);
        Assert.Equal(new[] { 17, 21 }, values);
        Assert.Equal(new[] { 19, 23 }, output);

        values = new GenericDerivedList<int> { 3 };
        readOnly = (IReadOnlyList<int>)values;
        fixture.Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { 13 }, values);
        Assert.Equal(new[] { 14, 23 }, output);
    }

    [Fact]
    public void InvalidSubclassIndexesStillRejectCopybackAndCachedCallsRecover()
    {
        IList<int> values = new DerivedList { 7 };
        int[] output = [41];
        int index = 1;
        Action<int> body = i =>
        {
            output[i] = 99;
            values[index] += 10;
        };

        Assert.Throws<ArgumentOutOfRangeException>(() => fixture.Gpu.For<int>(0, 1, body));
        Assert.Equal(new[] { 7 }, values);
        Assert.Equal(new[] { 41 }, output);

        index = 0;
        fixture.Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { 17 }, values);
        Assert.Equal(new[] { 99 }, output);
    }

    private sealed class DerivedList : List<int>;
    private sealed class GenericDerivedList<T> : List<T>;
}
