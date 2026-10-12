using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ReferenceFlowAuditRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void LocalArrayDescriptorReferencesCanBeReadAndReassigned()
    {
        int[] first = [3, 5], second = [7, 11];
        int[] output = new int[2], identities = new int[2];
        fixture.Gpu.For(0, output.Length, i =>
        {
            int[] selected = first;
            ref int[] alias = ref selected;
            output[i] = alias[i];
            alias = second;
            output[i] += selected[i];
            identities[i] = alias == second ? 1 : 0;
        });
        Assert.Equal(new[] { 10, 16 }, output);
        Assert.Equal(new[] { 1, 1 }, identities);
        Assert.Equal(new[] { 3, 5 }, first);
        Assert.Equal(new[] { 7, 11 }, second);
        VerifyFollowingLoop(output);
    }

    [Fact]
    public void HelpersCanReadAndReplaceLocalCollectionDescriptorsByReference()
    {
        IReadOnlyList<int>? missing = null;
        int[] first = [13, 17], second = [19, 23];
        int[] output = new int[2], nulls = new int[2];
        fixture.Gpu.For(0, output.Length, i =>
        {
            IReadOnlyList<int>? selected = missing;
            nulls[i] = IsMissing(in selected) ? 1 : 0;
            Replace(ref selected, first);
            output[i] = selected![i];
            Assign(out selected, second);
            output[i] += selected![i];
        });
        Assert.Equal(new[] { 1, 1 }, nulls);
        Assert.Equal(new[] { 32, 40 }, output);
        VerifyFollowingLoop(output);
    }

    [Fact]
    public void NullCoalescingAssignmentThroughReturnedReferencesPreservesTheSelectedArray()
    {
        int[] values = [41, 43], output = new int[2];
        fixture.Gpu.For(0, output.Length, i =>
        {
            int[]? selected = null;
            ref int[]? alias = ref Descriptor(ref selected);
            output[i] = (alias ??= values)[i];
            output[i] += (alias ??= values)[i];
        });
        Assert.Equal(new[] { 82, 86 }, output);
        VerifyFollowingLoop(output);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void InvalidAccessThroughDescriptorReferencesDefersErrorsAndCachedCallsRecover(bool missing)
    {
        int[]? input = missing ? null : [];
        int[] output = [47, 53];
        Action<int> body = i =>
        {
            int[]? selected = input;
            output[i] = 99;
            output[i] = Read(ref selected, i);
        };
        Exception? error = Record.Exception(() => fixture.Gpu.For(0, output.Length, body));
        Assert.IsType(missing ? typeof(NullReferenceException) : typeof(IndexOutOfRangeException), error);
        Assert.Equal(new[] { 47, 53 }, output);
        input = [59, 61];
        fixture.Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 59, 61 }, output);
        VerifyFollowingLoop(output);
    }

    [Fact]
    public void ReturnedDescriptorReferencesPreserveNullChecksAndLocalAliases()
    {
        int[]? missing = null;
        int[] values = [29, 31], output = new int[2];
        fixture.Gpu.For(0, output.Length, i =>
        {
            int[]? selected = missing;
            ref int[]? alias = ref Descriptor(ref selected);
            output[i] = alias is null ? 1 : 0;
            alias = values;
            output[i] += selected![i];
        });
        Assert.Equal(new[] { 30, 32 }, output);
        VerifyFollowingLoop(output);
    }

    private void VerifyFollowingLoop(int[] output)
    {
        fixture.Gpu.For(0, output.Length, i => output[i] = i + 37);
        Assert.Equal(new[] { 37, 38 }, output);
    }

    private static bool IsMissing(in IReadOnlyList<int>? collection) => collection is null;
    private static void Replace(ref IReadOnlyList<int>? target, IReadOnlyList<int> replacement) => target = replacement;
    private static void Assign(out IReadOnlyList<int>? target, IReadOnlyList<int> replacement) => target = replacement;
    private static ref int[]? Descriptor(ref int[]? collection) => ref collection;
    private static int Read(ref int[]? collection, int index) => collection![index];
}
