using System.Reflection;

namespace ParallelGpu;

/// <summary>Maps a captured host value to a field in the generated kernel state.</summary>
/// <param name="Path">The field path from the current closure or a static root to the captured value.</param>
/// <param name="Destination">The generated state field that receives the scalar or device view.</param>
/// <param name="Element">The supported collection element type, or null for a scalar.</param>
internal sealed record LoopCapture(FieldInfo[] Path, FieldInfo Destination, Type? Element)
{
    /// <summary>Reads the current captured value by following the original field path.</summary>
    /// <param name="target">The current delegate closure, or null for a static body.</param>
    /// <returns>The non-null captured value.</returns>
    /// <exception cref="ArgumentException">A captured object or collection is null.</exception>
    internal object Read(object? target)
    {
        object? value = target;
        foreach (var field in Path)
            value = field.GetValue(field.IsStatic ? null : value ??
                throw new ArgumentException("A captured object is null.", "body"));
        return value ?? throw new ArgumentException(
            $"Captured collection '{Path[^1].Name}' is null.", "body");
    }
}
