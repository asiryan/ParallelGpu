using System.Reflection;
using System.Reflection.Emit;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ClosureTranslationAuditTests(GpuFixture fixture)
{
    private static int[] boundDestination = [];

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ClosedStaticDelegatesAreRejectedBeforeBoundArgumentsCanBecomeLoopIndices(bool generic)
    {
        int[] output = [41];
        boundDestination = output;
        var method = typeof(ClosureTranslationAuditTests).GetMethod(nameof(BoundBody),
            BindingFlags.NonPublic | BindingFlags.Static)!;
        var body = (Action<int>)method.CreateDelegate(typeof(Action<int>), new object());
        // A closed static delegate binds marker; its exposed argument is the second parameter.
        body(0);
        Assert.Equal(123, output[0]);
        output[0] = 41;

        Assert.Throws<NotSupportedException>(() =>
        {
            if (generic) fixture.Gpu.For<int>(0, 1, body);
            else fixture.Gpu.For(0, 1, body);
        });
        Assert.Equal(41, output[0]);
        fixture.Gpu.For(0, 1, i => output[i] = i + 17);
        Assert.Equal(17, output[0]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DynamicMethodsWithoutInspectableManagedIlReportUnsupportedBodies(bool generic)
    {
        var method = new DynamicMethod("UnavailableLoopIl", typeof(void), [typeof(int)]);
        method.GetILGenerator().Emit(OpCodes.Ret);
        var body = (Action<int>)method.CreateDelegate(typeof(Action<int>));
        body(0);
        Assert.Throws<NotSupportedException>(() =>
        {
            if (generic) fixture.Gpu.For<int>(0, 1, body);
            else fixture.Gpu.For(0, 1, body);
        });
        int[] output = [41];
        fixture.Gpu.For(0, 1, i => output[i] = i + 19);
        Assert.Equal(19, output[0]);
    }

    [Fact]
    public void EmptyAndReversedRangesSkipUnsupportedDelegateInspection()
    {
        int[] output = [41];
        boundDestination = output;
        var closed = (Action<int>)typeof(ClosureTranslationAuditTests).GetMethod(nameof(BoundBody),
            BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate(typeof(Action<int>), new object());
        var dynamicMethod = new DynamicMethod("SkippedUnavailableLoopIl", typeof(void), [typeof(int)]);
        dynamicMethod.GetILGenerator().Emit(OpCodes.Ret);
        var dynamicBody = (Action<int>)dynamicMethod.CreateDelegate(typeof(Action<int>));

        foreach (var body in new[] { closed, dynamicBody })
        {
            fixture.Gpu.For(0, 0, body);
            fixture.Gpu.For(1, 0, body);
            fixture.Gpu.For<int>(0, 0, body);
            fixture.Gpu.For<int>(1, 0, body);
        }
        Assert.Equal(41, output[0]);
    }

    [Fact]
    public void CapturingLocalFunctionDelegatesRefreshTheirValuesAndCollections()
    {
        int[] input = [2, 3], output = [41, 43];
        int bias = 5;
        void Body(int i) => output[i] = input[i] + bias;
        Action<int> body = Body;

        fixture.Gpu.For(0, output.Length, body);
        Assert.Equal(new[] { 7, 8 }, output);
        int[] previousOutput = output;
        input = [11, 13];
        output = [47, 53];
        bias = 9;
        fixture.Gpu.For<int>(0, output.Length, body);
        Assert.Equal(new[] { 20, 22 }, output);
        Assert.Equal(new[] { 7, 8 }, previousOutput);
    }

    private static void BoundBody(object marker, int ignored)
    {
        if (marker != null)
            boundDestination[0] = 123;
    }
}
