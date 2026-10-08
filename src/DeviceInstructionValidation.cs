using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Rejects unsupported managed effects before they reach the device compiler.</summary>
internal static class DeviceInstructionValidation
{
    internal static void Validate(LoopInstruction instruction)
    {
        var code = instruction.Code;
        if (code.Name!.Contains(".ovf", StringComparison.Ordinal))
            throw new NotSupportedException($"Checked instruction '{code.Name}' is not supported in automatic GPU loops. Use unchecked arithmetic or explicit device-side range validation.");
        if (code == OpCodes.Newarr || code == OpCodes.Box || code == OpCodes.Unbox ||
            code == OpCodes.Unbox_Any || code == OpCodes.Castclass || code == OpCodes.Isinst ||
            code == OpCodes.Throw || code == OpCodes.Rethrow || code == OpCodes.Stsfld ||
            code == OpCodes.Ldsflda || code == OpCodes.Ldftn || code == OpCodes.Ldvirtftn ||
            code == OpCodes.Ldtoken || code == OpCodes.Calli || code == OpCodes.Constrained)
            throw new NotSupportedException($"Instruction '{code.Name}' requires managed runtime behavior inside a GPU helper. Use explicit GPU storage and device-compatible value operations.");
    }

    internal static void ValidateCallGraph(MethodBase method, HashSet<MethodBase> visited)
    {
        if (!visited.Add(method) || method.Module.Assembly == typeof(ILGPU.Atomic).Assembly)
            return;
        if (method is MethodInfo info && (!ScalarMethodTranslator.IsCompatible(info.ReturnType) ||
            info.GetParameters().Any(parameter => !ScalarMethodTranslator.IsCompatible(parameter.ParameterType))))
            throw new NotSupportedException($"Helper '{method}' uses managed parameters or a managed return value.");
        DeviceNumericMethods.Validate(method);
        // Supported framework operations are lowered by the collection/value translator or ILGPU.
        if (method.Module.Assembly == typeof(object).Assembly)
            return;
        if (method.GetMethodBody() is null)
            return;
        foreach (var instruction in LoopIL.Read(method))
        {
            Validate(instruction);
            if (instruction.Operand is FieldInfo field && field.IsStatic &&
                CapturedCollection.ElementType(field.FieldType) is not null)
                throw new NotSupportedException($"GPU helper '{method.Name}' captures static collection '{field.Name}'. Pass the collection as an explicit helper parameter.");
            if (instruction.Operand is FieldInfo mutable && mutable.IsStatic && !mutable.IsInitOnly && !mutable.IsLiteral)
                throw new NotSupportedException($"GPU helper '{method.Name}' reads mutable static field '{mutable.Name}'. Pass its current value as an explicit helper parameter.");
            if (instruction.Operand is MethodBase called)
                ValidateCallGraph(called, visited);
        }
    }
}
