using System.Reflection;

namespace ParallelGpu;

/// <summary>Selects framework math methods backed by native instructions or explicit lowering.</summary>
// Intrinsic signatures follow RemappedIntrinsics.Generated.tt at ILGPU commit
// 2efd942cf020da83c32278a6666d57c1d8505077 (the pinned 1.5.3 package).
internal static class DeviceNumericMethods
{
    internal static bool IsMath(MethodBase method) =>
        method.DeclaringType == typeof(Math) || method.DeclaringType == typeof(MathF);

    internal static bool IsLowered(MethodBase method) => IsExtendedMath(method) || method is MethodInfo info &&
        ((info.DeclaringType == typeof(Math) &&
          ((info.Name == nameof(Math.Clamp) && Matches(info, 3) && IsNumber(info.ReturnType)) ||
           (info.Name == nameof(Math.Abs) && Matches(info, 1) && IsSignedInteger(info.ReturnType)))) ||
         (IsMath(info) && info.Name is nameof(Math.Min) or nameof(Math.Max) && Matches(info, 2) &&
          (info.ReturnType == typeof(float) || info.ReturnType == typeof(double))) ||
         (IsMath(info) && info.Name == "CopySign" && Matches(info, 2) &&
          (info.ReturnType == typeof(float) || info.ReturnType == typeof(double))));

    internal static bool IsExtendedMath(MethodBase method) =>
        method is MethodInfo info && ExtendedMathReplacement(info) is not null;

    /// <summary>Resolves an exact framework signature to its internal device implementation.</summary>
    /// <remarks>Matching the complete parameter and return types keeps overloads for unsupported
    /// numeric representations outside the device contract. The context is supplied by the caller.</remarks>
    internal static MethodInfo? ExtendedMathReplacement(MethodInfo method)
    {
        if (!IsMath(method) || !method.IsStatic || method.IsGenericMethod)
            return null;
        var parameters = method.GetParameters().Select(parameter => parameter.ParameterType).ToArray();
        if (parameters.Length == 0)
            return null;
        string? suffix = TypeSuffix(parameters[0]);
        if (suffix is null)
            return null;
        var replacement = typeof(DeviceExtendedMath).GetMethod(method.Name + suffix,
            BindingFlags.Public | BindingFlags.Static, null,
            parameters.Append(typeof(DeviceExecutionContext)).ToArray(), null);
        return replacement?.ReturnType == method.ReturnType ? replacement : null;
    }

    /// <summary>Names the primitive device representation used in internal numeric helpers.</summary>
    private static string? TypeSuffix(Type type) => type == typeof(float) ? "Single" :
        type == typeof(double) ? "Double" : type == typeof(sbyte) ? "Int8" :
        type == typeof(byte) ? "UInt8" : type == typeof(short) ? "Int16" :
        type == typeof(ushort) ? "UInt16" : type == typeof(int) ? "Int32" :
        type == typeof(uint) ? "UInt32" : type == typeof(long) ? "Int64" :
        type == typeof(ulong) ? "UInt64" : null;

    internal static void Validate(MethodBase method)
    {
        if (IsMath(method) && !IsLowered(method) && !IsIntrinsic(method))
            throw new NotSupportedException($"Framework math method '{method}' is not supported by this GPU backend contract.");
    }

    private static bool IsIntrinsic(MethodBase method)
    {
        if (method is not MethodInfo info || !info.IsStatic || info.IsGenericMethod)
            return false;
        var type = info.ReturnType;
        if (info.DeclaringType == typeof(Math) && info.Name is nameof(Math.Min) or nameof(Math.Max) &&
            IsNumber(type) && Matches(info, 2))
            return true;
        var floating = info.DeclaringType == typeof(Math) ? typeof(double) : typeof(float);
        if (info.DeclaringType == typeof(Math) && info.Name == nameof(Math.Abs) &&
            type == typeof(float) && Matches(info, 1))
            return true;
        if (type != floating)
            return false;
        // This native subset needs neither Algorithms lowering nor a configured CUDA LibDevice.
        return info.Name switch
        {
            nameof(Math.Abs) or nameof(Math.Ceiling) or nameof(Math.Floor) or nameof(Math.Sqrt) => Matches(info, 1),
            nameof(Math.Min) or nameof(Math.Max) => Matches(info, 2),
            _ => false,
        };
    }
    private static bool Matches(MethodInfo method, int count)
    {
        var parameters = method.GetParameters();
        return method.IsStatic && !method.IsGenericMethod && parameters.Length == count &&
            parameters.All(parameter => parameter.ParameterType == method.ReturnType);
    }

    private static bool IsSignedInteger(Type type) => type == typeof(sbyte) || type == typeof(short) ||
        type == typeof(int) || type == typeof(long);

    private static bool IsNumber(Type type) => IsSignedInteger(type) || type == typeof(byte) ||
        type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong) ||
        type == typeof(float) || type == typeof(double);
}
