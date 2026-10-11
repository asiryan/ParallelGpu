namespace ParallelGpu;

/// <summary>Recognizes native append and object-graph views and the device execution context.</summary>
internal static class DeviceNativeTypes
{
    internal static bool IsSupported(Type type) => type == typeof(DeviceExecutionContext) ||
        type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(GpuAppendView<>) ||
            type.GetGenericTypeDefinition() == typeof(GpuObjectGraphView<>));

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
