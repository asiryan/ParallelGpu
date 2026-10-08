namespace ParallelGpu;

/// <summary>Describes a supported captured host collection and its translated device view.</summary>
internal abstract class CapturedCollection
{
    /// <summary>Gets the device representation after the invocation's buffers have been uploaded.</summary>
    internal abstract DeviceCollection View { get; }

    /// <summary>Checks whether a type is supported as a collection element.</summary>
    /// <param name="type">The candidate element type.</param>
    /// <returns>True for supported numeric, Boolean, character, and enum types; otherwise, false.</returns>
    internal static bool IsSupportedElementType(Type type) => type == typeof(float) || type == typeof(int) ||
        type == typeof(uint) || type == typeof(long) || type == typeof(ulong) ||
        type == typeof(double) || type == typeof(short) || type == typeof(ushort) ||
        type == typeof(byte) || type == typeof(sbyte) || type == typeof(bool) || type == typeof(char) || type.IsEnum;

    /// <summary>Finds the element type of a supported array or list declaration.</summary>
    /// <param name="type">The declared collection type.</param>
    /// <returns>The element type, or null if the collection declaration is unsupported.</returns>
    internal static Type? ElementType(Type type)
    {
        if (type.IsArray)
        {
            var element = type.GetElementType()!;
            if (type.GetArrayRank() <= 2 && IsSupportedElementType(element))
                return element;
            if (type.GetArrayRank() == 1 && element.IsArray && element.GetArrayRank() == 1 &&
                IsSupportedElementType(element.GetElementType()!))
                return element.GetElementType();
        }
        if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(List<>) ||
            type.GetGenericTypeDefinition() == typeof(IList<>) ||
            type.GetGenericTypeDefinition() == typeof(IReadOnlyList<>)) &&
            IsSupportedElementType(type.GetGenericArguments()[0]))
            return type.GetGenericArguments()[0];
        return null;
    }
}
