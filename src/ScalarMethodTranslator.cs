using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Maps helper methods and constructors to the device representations of their value types.</summary>
/// <remarks>Compatible methods retain their original implementation unless a signature, local,
/// field, or transitive call needs character or structure translation.</remarks>
internal sealed class ScalarMethodTranslator
{
    /// <summary>The generated type containing translated static helpers.</summary>
    private readonly TypeBuilder helpers;
    /// <summary>Original methods and constructors mapped to their static device helpers.</summary>
    private readonly Dictionary<MethodBase, MethodInfo> methods = new();
    /// <summary>Readonly static values whose representation changes inside translated helpers.</summary>
    private readonly Dictionary<FieldInfo, FieldBuilder> staticValues = new();
    private readonly CollectionMethodTranslator collections;
    private readonly HashSet<MethodBase> validated = new();

    /// <summary>Creates a helper type in the same module as the generated loop.</summary>
    /// <param name="module">The dynamic module owning the loop and its helpers.</param>
    internal ScalarMethodTranslator(ModuleBuilder module)
    {
        helpers = module.DefineType("ScalarHelpers",
            TypeAttributes.Public | TypeAttributes.Abstract | TypeAttributes.Sealed);
        collections = new CollectionMethodTranslator(helpers);
    }

    /// <summary>Maps full-width character values and device-compatible structures, including references.</summary>
    /// <param name="type">The original value or reference type.</param>
    /// <returns>The corresponding device-compatible type.</returns>
    internal static Type MapType(Type type)
    {
        if (type.IsByRef)
            return MapType(type.GetElementType()!).MakeByRefType();
        if (DeviceNativeTypes.IsSupported(type))
        {
            DeviceNativeTypes.Validate(type);
            return type;
        }
        return CapturedCollection.ElementType(type) is not null
            ? typeof(DeviceCollection) : DeviceValueType.MapType(type);
    }

    internal static bool IsCompatible(Type type) => type == typeof(void) ||
        DeviceValueType.IsSupported(type.IsByRef ? type.GetElementType()! : type) ||
        DeviceNativeTypes.IsSupported(type.IsByRef ? type.GetElementType()! : type) ||
        CapturedCollection.ElementType(type.IsByRef ? type.GetElementType()! : type) is not null;

    internal void EmitCall(ILGenerator il, MethodInfo method, Action<ILGenerator> emitContext)
    {
        if (collections.TryEmit(il, method))
            return;
        if ((!method.IsStatic && !IsCompatible(method.DeclaringType!)) ||
            !IsCompatible(method.ReturnType) || method.GetParameters().Any(p => !IsCompatible(p.ParameterType)))
            throw new NotSupportedException($"Method '{method.DeclaringType!.FullName}.{method.Name}' is not device-compatible.");
        var replacement = Translate(method);
        if (replacement != method)
            emitContext(il);
        il.Emit(OpCodes.Call, replacement);
    }

    /// <summary>Gets a method using the device representations of its values.</summary>
    /// <param name="method">The original method.</param>
    /// <returns>A generated helper or the unchanged method when translation is unnecessary.</returns>
    /// <exception cref="NotSupportedException">A translated helper uses unsupported managed IL.</exception>
    internal MethodInfo Translate(MethodInfo method)
    {
        if (methods.TryGetValue(method, out var existing))
            return existing;
        if (method.Module.Assembly == typeof(ILGPU.Atomic).Assembly)
            return method;
        DeviceInstructionValidation.ValidateCallGraph(method, validated);
        if (RequiresTranslation(method, new HashSet<MethodBase>()))
            return TranslateCore(method);
        methods.Add(method, method);
        return method;
    }

    /// <summary>Emits a compatible value constructor, translating it when its representation changes.</summary>
    /// <param name="il">The destination IL generator.</param>
    /// <param name="code">The original call or construction instruction.</param>
    /// <param name="constructor">The value constructor.</param>
    /// <param name="emitContext">Emits the current invocation error context.</param>
    /// <exception cref="NotSupportedException">The constructor uses managed objects or unsupported parameters.</exception>
    internal void EmitConstructor(ILGenerator il, OpCode code, ConstructorInfo constructor, Action<ILGenerator> emitContext)
    {
        if (collections.TryEmitConstructor(il, code, constructor))
            return;
        var parameters = constructor.GetParameters();
        if (!DeviceValueType.IsSupported(constructor.DeclaringType!) ||
            parameters.Any(parameter => !IsCompatible(parameter.ParameterType)))
            throw new NotSupportedException($"Constructor '{constructor.DeclaringType}' is not supported inside GPU loops.");
        DeviceInstructionValidation.ValidateCallGraph(constructor, validated);
        if (!RequiresTranslation(constructor, new HashSet<MethodBase>()))
        {
            il.Emit(code, constructor);
            return;
        }
        var replacement = TranslateCore(constructor);
        if (code != OpCodes.Newobj)
        {
            emitContext(il);
            il.Emit(OpCodes.Call, replacement);
            return;
        }
        // Constructor arguments are already on the stack; save them before loading the new value's address.
        var arguments = parameters.Select(parameter => il.DeclareLocal(MapType(parameter.ParameterType))).ToArray();
        for (int i = arguments.Length - 1; i >= 0; i--)
            il.Emit(OpCodes.Stloc, arguments[i]);
        var value = il.DeclareLocal(MapType(constructor.DeclaringType!));
        il.Emit(OpCodes.Ldloca, value);
        il.Emit(OpCodes.Initobj, value.LocalType);
        il.Emit(OpCodes.Ldloca, value);
        foreach (var argument in arguments)
            il.Emit(OpCodes.Ldloc, argument);
        emitContext(il);
        il.Emit(OpCodes.Call, replacement);
        il.Emit(OpCodes.Ldloc, value);
    }

    /// <summary>Copies a method or constructor into a static helper with mapped signatures and instructions.</summary>
    /// <param name="method">The original helper or value constructor.</param>
    /// <returns>The generated static helper.</returns>
    private MethodInfo TranslateCore(MethodBase method)
    {
        if (methods.TryGetValue(method, out var existing))
            return existing;
        var parameters = method.GetParameters().Select(parameter => MapType(parameter.ParameterType));
        if (!method.IsStatic)
            parameters = parameters.Prepend(MapType(method.DeclaringType!.MakeByRefType()));
        int contextArgument = parameters.Count();
        parameters = parameters.Append(typeof(DeviceExecutionContext));
        var returnType = method is MethodInfo info ? MapType(info.ReturnType) : typeof(void);
        var replacement = helpers.DefineMethod("Value" + methods.Count,
            MethodAttributes.Public | MethodAttributes.Static, returnType, parameters.ToArray());
        // Register before translating calls so recursive references resolve to the same generated method.
        methods.Add(method, replacement);
        var il = replacement.GetILGenerator();
        Action<ILGenerator> emitContext = target => target.Emit(OpCodes.Ldarg, (short)contextArgument);
        if (method is MethodInfo nullableValue && nullableValue.Name == "get_Value" &&
            Nullable.GetUnderlyingType(method.DeclaringType!) is not null)
        {
            EmitNullableValue(il, nullableValue, emitContext);
            return replacement;
        }
        foreach (var local in method.GetMethodBody()!.LocalVariables)
        {
            if (local.IsPinned)
                throw new NotSupportedException("Pinned variables are not supported in GPU value helpers.");
            il.DeclareLocal(MapType(local.LocalType));
        }
        var instructions = LoopIL.Read(method);
        var stackTypes = LoopStackTypes.Analyze(method, instructions);
        var labels = instructions.ToDictionary(instruction => instruction.Offset, _ => il.DefineLabel());
        foreach (var instruction in instructions)
        {
            il.MarkLabel(labels[instruction.Offset]);
            if (instruction.Code == OpCodes.Initobj && instruction.Operand is Type initialized &&
                initialized.IsValueType && CapturedCollection.ElementType(initialized) is not null)
            {
                emitContext(il);
                il.Emit(OpCodes.Call, typeof(DeviceCollectionSlice).GetMethod(nameof(DeviceCollectionSlice.Initialize))!);
                continue;
            }
            if (stackTypes.TryGetValue(instruction.Offset, out var stack) &&
                LoopStackTypes.TryEmitReferenceOperation(il, instruction, stack, labels, emitContext))
                continue;
            if (LambdaKernel.TryArrayOperation(il, instruction))
                continue;
            if (instruction.Operand is ConstructorInfo constructor &&
                (instruction.Code == OpCodes.Call || instruction.Code == OpCodes.Newobj))
            {
                EmitConstructor(il, instruction.Code, constructor, emitContext);
                continue;
            }
            if (instruction.Operand is MethodInfo called &&
                (instruction.Code == OpCodes.Call || instruction.Code == OpCodes.Callvirt))
            {
                EmitCall(il, called, emitContext);
                continue;
            }
            DeviceInstructionValidation.Validate(instruction);
            if (instruction.Operand is FieldInfo field)
            {
                if (field.DeclaringType == typeof(char))
                {
                    // A character instance is now a UInt16 reference rather than a managed Char value.
                    if (instruction.Code == OpCodes.Ldfld)
                        il.Emit(OpCodes.Ldind_U2);
                    else if (instruction.Code != OpCodes.Ldflda)
                        throw new NotSupportedException("Character field mutation is not supported in GPU value helpers.");
                }
                else if (field.IsStatic && MapType(field.FieldType) != field.FieldType)
                {
                    if (instruction.Code != OpCodes.Ldsfld || !field.IsInitOnly)
                        throw new NotSupportedException("Mapped static helper values must be readonly.");
                    if (!staticValues.TryGetValue(field, out var mapped))
                    {
                        mapped = helpers.DefineField("StaticValue" + staticValues.Count, MapType(field.FieldType),
                            FieldAttributes.Public | FieldAttributes.Static | FieldAttributes.InitOnly);
                        staticValues.Add(field, mapped);
                    }
                    il.Emit(OpCodes.Ldsfld, mapped);
                }
                else
                    il.Emit(instruction.Code, DeviceValueType.MapField(field));
                continue;
            }
            LoopIL.Emit(il, instruction.Operand is Type type
                ? instruction with { Operand = MapType(type) } : instruction, labels);
        }
        return replacement;
    }

    private static void EmitNullableValue(ILGenerator il, MethodInfo method, Action<ILGenerator> emitContext)
    {
        var fields = method.DeclaringType!.GetFields(BindingFlags.Instance | BindingFlags.NonPublic);
        var present = fields.Single(field => field.FieldType == typeof(bool));
        var value = fields.Single(field => field.Name == "value");
        var available = il.DefineLabel();
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldfld, DeviceValueType.MapField(present));
        il.Emit(OpCodes.Brtrue, available);
        emitContext(il);
        var context = il.DeclareLocal(typeof(DeviceExecutionContext));
        il.Emit(OpCodes.Stloc, context);
        il.Emit(OpCodes.Ldloca, context);
        il.Emit(OpCodes.Ldc_I4_1);
        il.Emit(OpCodes.Call, typeof(DeviceExecutionContext).GetMethod(nameof(DeviceExecutionContext.Fail))!);
        il.MarkLabel(available);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldfld, DeviceValueType.MapField(value));
        il.Emit(OpCodes.Ret);
    }
    /// <summary>Checks whether a signature, local, field, or transitive call needs value translation.</summary>
    /// <param name="method">The helper to inspect.</param>
    /// <param name="visited">The methods already inspected along the current call graph.</param>
    /// <returns>True when character or structure translation is required.</returns>
    private static bool RequiresTranslation(MethodBase method, HashSet<MethodBase> visited)
    {
        if (method.Module.Assembly == typeof(ILGPU.Atomic).Assembly || !visited.Add(method))
            return false;
        if ((!method.IsStatic && MapType(method.DeclaringType!) != method.DeclaringType) ||
            (method is MethodInfo info && MapType(info.ReturnType) != info.ReturnType) ||
            method.GetParameters().Any(parameter => MapType(parameter.ParameterType) != parameter.ParameterType))
            return true;
        var body = method.GetMethodBody();
        if (body is null)
            return false;
        if (body.LocalVariables.Any(local => MapType(local.LocalType) != local.LocalType))
            return true;
        return body.ExceptionHandlingClauses.Count == 0 &&
            LoopIL.Read(method).Any(instruction =>
                (instruction.Operand is MethodBase called && RequiresTranslation(called, visited)) ||
                (instruction.Operand is FieldInfo field &&
                    (DeviceValueType.MapField(field) != field || MapType(field.FieldType) != field.FieldType)));
    }

    /// <summary>Finalizes generated helpers before the loop entry point is loaded by ILGPU.</summary>
    internal void Complete()
    {
        if (staticValues.Count > 0)
        {
            // ILGPU reads readonly static values at compilation; initialize their mirrors on the host.
            var il = helpers.DefineTypeInitializer().GetILGenerator();
            foreach (var pair in staticValues)
            {
                il.Emit(OpCodes.Ldsfld, pair.Key);
                il.Emit(OpCodes.Box, pair.Key.FieldType);
                il.Emit(OpCodes.Ldtoken, pair.Key.FieldType);
                il.Emit(OpCodes.Call, typeof(Type).GetMethod(nameof(Type.GetTypeFromHandle))!);
                il.Emit(OpCodes.Call, typeof(DeviceValueType).GetMethod(nameof(DeviceValueType.ToDevice),
                    BindingFlags.NonPublic | BindingFlags.Static)!);
                il.Emit(OpCodes.Unbox_Any, pair.Value.FieldType);
                il.Emit(OpCodes.Stsfld, pair.Value);
            }
            il.Emit(OpCodes.Ret);
        }
        helpers.CreateType();
    }
}
