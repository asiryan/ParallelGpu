using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class ReleaseTranslatorRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void InlineMixedCollectionConditionsPreserveIdentityNullAndInterfaceAccess()
    {
        int[] first = [2, 3, 5, 7];
        List<int> second = [11, 13, 17, 19];
        IReadOnlyList<int>? missing = null;
        int[] values = new int[4], identity = new int[4], nulls = new int[4];
        fixture.Gpu.For(0, values.Length, i =>
        {
            values[i] = (i % 2 == 0 ? first : (IReadOnlyList<int>)second)[i];
            identity[i] = (i % 2 == 0 ? first : (IReadOnlyList<int>)second) == first ? 1 : 0;
            nulls[i] = (i % 2 == 0 ? first : missing) == null ? 1 : 0;
        });
        Assert.Equal(new[] { 2, 13, 5, 19 }, values);
        Assert.Equal(new[] { 1, 0, 1, 0 }, identity);
        Assert.Equal(new[] { 0, 1, 0, 1 }, nulls);
    }

    [Fact]
    public void CachedHelpersComposeMappedValuesMathTuplesAndDeferredNullableErrors()
    {
        var settings = new ProjectionSettings('A', false);
        double[] angles = [0.25, 0.75];
        bool[] enabled = [true, true];
        char[] markers = ['?', '?'];
        double[] values = [-1, -1];
        Action<int> body = i =>
        {
            var projected = Project(settings, angles[i]);
            enabled[i] = projected.Enabled;
            markers[i] = projected.Marker;
            values[i] = projected.Value;
        };
        fixture.Gpu.For(0, angles.Length, body);
        Assert.Equal(new[] { false, false }, enabled);
        Assert.Equal(new[] { 'B', 'B' }, markers);
        for (int i = 0; i < angles.Length; i++)
            Assert.InRange(Math.Abs(values[i] - (Math.Sin(angles[i]) + Math.Cos(angles[i]))), 0, 4e-12);

        settings = new ProjectionSettings('\uffff', true);
        fixture.Gpu.For(0, angles.Length, body);
        Assert.Equal(new[] { true, true }, enabled);
        Assert.Equal(new[] { '\0', '\0' }, markers);
        double[] beforeFailure = (double[])values.Clone();
        settings = new ProjectionSettings('X', null);
        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, angles.Length, body));
        Assert.Equal(new[] { true, true }, enabled);
        Assert.Equal(new[] { '\0', '\0' }, markers);
        Assert.Equal(beforeFailure, values);

        settings = new ProjectionSettings('Z', false);
        fixture.Gpu.For(0, angles.Length, body);
        Assert.Equal(new[] { false, false }, enabled);
        Assert.Equal(new[] { '[', '[' }, markers);
    }

    private static (bool Enabled, char Marker, double Value) Project(ProjectionSettings settings, double angle)
    {
        var trigonometry = Math.SinCos(angle);
        return (settings.Enabled!.Value, unchecked((char)(settings.Marker + 1)), trigonometry.Sin + trigonometry.Cos);
    }

    private readonly struct ProjectionSettings(char marker, bool? enabled)
    {
        public readonly char Marker = marker;
        public readonly bool? Enabled = enabled;
    }
}
