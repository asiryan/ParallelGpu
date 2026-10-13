using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Validates helper signatures and rejects unsupported IL instructions and static captures.</summary>
internal static class DeviceInstructionValidation
{
    /// <summary>Rejects checked arithmetic and instructions requiring managed runtime behavior.</summary>
    /// <param name="instruction">The original loop or helper instruction.</param>
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

    /// <summary>Recognizes readonly scalar addresses that helpers can consume through a local value copy.</summary>
    /// <param name="instruction">The original static-field address load.</param>
    /// <returns>True for supported readonly scalar values, excluding collections and native device views.</returns>
    internal static bool IsReadOnlyScalarAddress(LoopInstruction instruction) =>
        instruction.Code == OpCodes.Ldsflda && instruction.Operand is FieldInfo field &&
        field.IsStatic && field.IsInitOnly && !ContainsNativeView(field.FieldType, new HashSet<Type>()) &&
        DeviceValueType.IsSupported(field.FieldType) && CapturedCollection.ElementType(field.FieldType) is null;

    /// <summary>Recognizes borrowed native views inside value wrappers without reading static values.</summary>
    /// <param name="type">The declared static field or nested instance field type.</param>
    /// <param name="visited">Types already inspected along the value layout.</param>
    /// <returns>True when a native view occurs directly or inside a value-type field.</returns>
    private static bool ContainsNativeView(Type type, HashSet<Type> visited)
    {
        if (DeviceNativeTypes.IsSupported(type))
            return true;
        if (!type.IsValueType || type.IsPrimitive || type.IsEnum || !visited.Add(type))
            return false;
        return type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Any(field => ContainsNativeView(field.FieldType, visited));
    }

    /// <summary>Validates transitive helper calls, leaving recognized framework and ILGPU operations to their translators.</summary>
    /// <param name="method">The method or constructor to inspect.</param>
    /// <param name="visited">The methods already inspected, including recursive references.</param>
    internal static void ValidateCallGraph(MethodBase method, HashSet<MethodBase> visited)
    {
        if (!visited.Add(method) || method.Module.Assembly == typeof(ILGPU.Atomic).Assembly)
            return;
        if (method is ConstructorInfo constructor &&
            (!constructor.DeclaringType!.IsValueType || !ScalarMethodTranslator.IsCompatible(constructor.DeclaringType) ||
             constructor.GetParameters().Any(parameter => !ScalarMethodTranslator.IsCompatible(parameter.ParameterType))))
            throw new NotSupportedException($"Constructor '{constructor}' is not a device-compatible value constructor. Managed allocation is unsupported inside GPU helpers.");
        if (method is MethodInfo info && (!ScalarMethodTranslator.IsCompatible(info.ReturnType) ||
            info.GetParameters().Any(parameter => !ScalarMethodTranslator.IsCompatible(parameter.ParameterType))))
            throw new NotSupportedException($"Helper '{method}' uses managed parameters or a managed return value.");
        DeviceNumericMethods.Validate(method);
        // Supported framework operations are lowered by the collection/value translator or ILGPU.
        if (method.Module.Assembly == typeof(object).Assembly)
            return;
        if (method.GetMethodBody() is null)
            return;
        var instructions = LoopIL.Read(method);
        for (int index = 0; index < instructions.Count; index++)
        {
            var instruction = instructions[index];
            if (instruction.Operand is FieldInfo native && native.IsStatic &&
                ContainsNativeView(native.FieldType, new HashSet<Type>()))
                throw new NotSupportedException($"GPU helper '{method.Name}' captures static native GPU view storage '{native.Name}'. Pass the view as an explicit helper parameter or capture it in the loop body.");
            if (IsReadOnlyScalarAddress(instruction))
                ReadOnlyCapture.Validate(instructions, index);
            else
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
