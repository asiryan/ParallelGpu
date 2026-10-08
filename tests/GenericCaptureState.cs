namespace ParallelGpu.Tests;

/// <summary>Supplies separate static capture values for each closed generic test type.</summary>
/// <typeparam name="T">The type distinguishing one set of static fields from another.</typeparam>
internal static class GenericCaptureState<T>
{
    /// <summary>The scalar read by the generic capture regression loop.</summary>
    internal static int Scalar;

    /// <summary>The array read and modified by the generic capture regression loop.</summary>
    internal static int[] Values = [];
}
