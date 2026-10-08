using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Preserves UTF-16 character values in scalar helper signatures and local variables.</summary>
/// <remarks>Only methods that use character values are copied; other scalar methods retain their original implementation.</remarks>
internal sealed class ScalarMethodTranslator
{
    /// <summary>The generated type containing translated static helpers.</summary>
    private readonly TypeBuilder helpers;
    /// <summary>Original methods mapped to their translated or unchanged implementations.</summary>
    private readonly Dictionary<MethodInfo, MethodInfo> methods = new();

    /// <summary>Creates a helper type in the same module as the generated loop.</summary>
    /// <param name="module">The dynamic module owning the loop and its scalar helpers.</param>
    internal ScalarMethodTranslator(ModuleBuilder module) =>
        helpers = module.DefineType("ScalarHelpers",
            TypeAttributes.Public | TypeAttributes.Abstract | TypeAttributes.Sealed);

    /// <summary>Maps character values and references to their full-width UInt16 representation.</summary>
    /// <param name="type">The original scalar or reference type.</param>
    /// <returns>The corresponding device-compatible type.</returns>
    internal static Type MapType(Type type) => type == typeof(char) ? typeof(ushort) :
        type == typeof(char).MakeByRefType() ? typeof(ushort).MakeByRefType() : type;

    /// <summary>Gets a scalar helper whose character values retain all UTF-16 bits.</summary>
    /// <param name="method">The original scalar method.</param>
    /// <returns>A generated helper or the unchanged method when translation is unnecessary.</returns>
    /// <exception cref="NotSupportedException">A character helper uses unsupported managed IL.</exception>
    internal MethodInfo Translate(MethodInfo method)
    {
        if (methods.TryGetValue(method, out var existing))
            return existing;
        if (!RequiresTranslation(method, new HashSet<MethodInfo>()))
        {
            methods.Add(method, method);
            return method;
        }

        var parameters = method.GetParameters().Select(parameter => MapType(parameter.ParameterType));
        if (!method.IsStatic)
            parameters = parameters.Prepend(MapType(method.DeclaringType!.MakeByRefType()));
        var replacement = helpers.DefineMethod("Scalar" + methods.Count,
            MethodAttributes.Public | MethodAttributes.Static, MapType(method.ReturnType), parameters.ToArray());
        // Register before translating calls so recursive references resolve to the same generated method.
        methods.Add(method, replacement);
        var il = replacement.GetILGenerator();
        foreach (var local in method.GetMethodBody()!.LocalVariables)
        {
            if (local.IsPinned)
                throw new NotSupportedException("Pinned variables are not supported in GPU scalar helpers.");
            il.DeclareLocal(MapType(local.LocalType));
        }
        var instructions = LoopIL.Read(method);
        var labels = instructions.ToDictionary(instruction => instruction.Offset, _ => il.DefineLabel());
        foreach (var instruction in instructions)
        {
            il.MarkLabel(labels[instruction.Offset]);
            if (instruction.Operand is MethodInfo called &&
                (instruction.Code == OpCodes.Call || instruction.Code == OpCodes.Callvirt))
            {
                il.Emit(OpCodes.Call, Translate(called));
                continue;
            }
            if (instruction.Operand is FieldInfo field && field.DeclaringType == typeof(char))
            {
                // A character instance is now a UInt16 reference rather than a managed Char value.
                if (instruction.Code == OpCodes.Ldfld)
                    il.Emit(OpCodes.Ldind_U2);
                else if (instruction.Code != OpCodes.Ldflda)
                    throw new NotSupportedException("Character field mutation is not supported in GPU scalar helpers.");
                continue;
            }
            LoopIL.Emit(il, instruction.Operand is Type type
                ? instruction with { Operand = MapType(type) } : instruction, labels);
        }
        return replacement;
    }

    /// <summary>Checks whether a method's signature, locals, or scalar calls use character values.</summary>
    /// <param name="method">The scalar helper to inspect.</param>
    /// <param name="visited">The methods already inspected along the current call graph.</param>
    /// <returns>True when character translation is required.</returns>
    private static bool RequiresTranslation(MethodInfo method, HashSet<MethodInfo> visited)
    {
        if (!visited.Add(method))
            return false;
        if (HasCharacterSignature(method))
            return true;
        var body = method.GetMethodBody();
        if (body is null)
            return false;
        if (body.LocalVariables.Any(local => MapType(local.LocalType) != local.LocalType))
            return true;
        return body.ExceptionHandlingClauses.Count == 0 &&
            LoopIL.Read(method).Any(instruction =>
                instruction.Operand is MethodInfo called && RequiresTranslation(called, visited));
    }

    /// <summary>Checks whether a scalar method receives or returns character values or references.</summary>
    /// <param name="method">The method whose signature is inspected.</param>
    /// <returns>True when the declaring type, return type, or a parameter requires character translation.</returns>
    private static bool HasCharacterSignature(MethodInfo method) =>
        method.DeclaringType == typeof(char) || MapType(method.ReturnType) != method.ReturnType ||
        method.GetParameters().Any(parameter => MapType(parameter.ParameterType) != parameter.ParameterType);

    /// <summary>Finalizes generated helpers before the loop entry point is loaded by ILGPU.</summary>
    internal void Complete() => helpers.CreateType();
}
