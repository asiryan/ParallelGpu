namespace ParallelGpu;

/// <summary>Describes a captured host collection and its translated device view.</summary>
internal abstract class CapturedCollection
{
    /// <summary>Gets the translated view after upload.</summary>
    internal abstract DeviceCollection View { get; }
    /// <summary>Gets the nested-array child descriptor.</summary>
    internal abstract RowDescriptor Row { get; }
    /// <summary>Gets or sets the invocation-local host reference identity.</summary>
    internal int Identity { get; set; }

    /// <summary>Checks whether a type is supported as a leaf element.</summary>
    /// <param name="type">The candidate element type.</param>
    /// <returns>Whether the element has a supported device representation.</returns>
    internal static bool IsSupportedElementType(Type type) => DeviceValueType.IsSupported(type);

    /// <summary>Finds a supported collection's ultimate leaf element type.</summary>
    /// <param name="type">The declared collection type.</param>
    /// <returns>The leaf element type, or null for unsupported collections.</returns>
    internal static Type? ElementType(Type type)
    {
        if (type.IsArray)
        {
            do { type = type.GetElementType()!; } while (type.IsArray);
            return IsSupportedElementType(type) ? type : null;
        }
        if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(List<>) ||
            type.GetGenericTypeDefinition() == typeof(IList<>) ||
            type.GetGenericTypeDefinition() == typeof(IReadOnlyList<>)) &&
            IsSupportedElementType(type.GetGenericArguments()[0]))
            return type.GetGenericArguments()[0];
        var adapted = GpuCollectionAdapters.ElementType(type);
        return adapted is not null && IsSupportedElementType(adapted) ? adapted : null;
    }
}
