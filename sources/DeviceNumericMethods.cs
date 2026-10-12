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

    internal static bool IsExtendedMath(MethodBase method) => method is MethodInfo info && IsMath(info) &&
        info.ReturnType == (info.DeclaringType == typeof(Math) ? typeof(double) : typeof(float)) &&
        (info.Name switch
        {
            nameof(Math.Sin) or nameof(Math.Cos) or nameof(Math.Exp) => Matches(info, 1),
            nameof(Math.Pow) => Matches(info, 2),
            nameof(Math.Log) => Matches(info, 1) || Matches(info, 2),
            _ => false,
        });

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
