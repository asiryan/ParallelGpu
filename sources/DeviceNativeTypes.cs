using System.Reflection;

namespace ParallelGpu;

/// <summary>Recognizes native buffer, append, and object-graph views and the device execution context.</summary>
internal static class DeviceNativeTypes
{
    internal static bool IsSupported(Type type) => type == typeof(DeviceExecutionContext) ||
        type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(GpuView<>) ||
            type.GetGenericTypeDefinition() == typeof(GpuAppendView<>) ||
            type.GetGenericTypeDefinition() == typeof(GpuObjectGraphView<>));

    /// <summary>Recognizes buffer element references that do not expose the captured view descriptor.</summary>
    /// <param name="method">The instance accessor returning a device element reference.</param>
    /// <returns>True for the ref-returning integer indexer of a native buffer view.</returns>
    internal static bool IsElementReference(MethodInfo method) =>
        method.DeclaringType is { IsGenericType: true } declaring &&
        declaring.GetGenericTypeDefinition() == typeof(GpuView<>) &&
        !method.IsStatic && method.Name == "get_Item" && method.ReturnType.IsByRef &&
        method.ReturnType.GetElementType() == declaring.GetGenericArguments()[0] &&
        method.GetParameters() is [{ ParameterType: var index }] && index == typeof(int);

    /// <summary>Checks that lambda value mapping cannot change an explicitly allocated native payload.</summary>
    /// <param name="type">A recognized native view or execution-context type.</param>
    /// <exception cref="NotSupportedException">A view payload is unsupported, not publicly visible, or requires value conversion.</exception>
    internal static void Validate(Type type)
    {
        if (!type.IsGenericType)
            return;
        var payload = type.GetGenericArguments()[0];
        if (!payload.IsVisible || !DeviceValueType.IsSupported(payload) || DeviceValueType.MapType(payload) != payload)
            throw new NotSupportedException(
                $"Native GPU view '{type.Name}' uses payload '{payload}' requiring automatic value conversion. " +
                "Use a public device representation with identity-mapped numeric fields (UInt16 for Char and Int32 for Boolean fields), " +
                "or use an explicit value kernel with its ILGPU-compatible representation.");
    }
}
