using System.Reflection;

namespace ParallelGpu;

/// <summary>Selects framework math methods backed by native instructions or explicit lowering.</summary>
// Intrinsic signatures follow RemappedIntrinsics.Generated.tt at ILGPU commit
// 2efd942cf020da83c32278a6666d57c1d8505077 (the pinned 1.5.3 package).
internal static class DeviceNumericMethods
{
    internal static bool IsMath(MethodBase method) =>
        method.DeclaringType == typeof(Math) || method.DeclaringType == typeof(MathF);

    internal static bool IsLowered(MethodBase method) => method is MethodInfo info &&
        ((info.DeclaringType == typeof(Math) &&
          ((info.Name == nameof(Math.Clamp) && Matches(info, 3) && IsNumber(info.ReturnType)) ||
           (info.Name == nameof(Math.Abs) && Matches(info, 1) && IsSignedInteger(info.ReturnType)))) ||
         (IsMath(info) && info.Name is nameof(Math.Min) or nameof(Math.Max) && Matches(info, 2) &&
          (info.ReturnType == typeof(float) || info.ReturnType == typeof(double))));

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
        // Use the native subset available without optional Algorithms or a configured CUDA LibDevice.
        return info.Name switch
        {
            nameof(Math.Abs) or nameof(Math.Ceiling) or nameof(Math.Floor) or nameof(Math.Sqrt) => Matches(info, 1),
            "CopySign" or nameof(Math.Min) or nameof(Math.Max) => Matches(info, 2),
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