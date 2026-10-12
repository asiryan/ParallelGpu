using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Maps helpers and constructors to device value and collection representations and guarded operations.</summary>
/// <remarks>Compatible methods retain their original implementation unless a signature, local,
/// field, transitive call, or numeric operation needs representation conversion or deferred error guards.</remarks>
internal sealed class ScalarMethodTranslator
{
    /// <summary>The generated type containing translated static helpers.</summary>
    private readonly TypeBuilder helpers;
    /// <summary>The generated assembly that hosts helpers and mapped static value initializers.</summary>
    private readonly AssemblyBuilder generatedAssembly;
    /// <summary>The transitive helper assemblies whose private fields may be read by host initializers.</summary>
    private readonly HashSet<string> accessAssemblies = new();
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
        generatedAssembly = (AssemblyBuilder)module.Assembly;
        helpers = module.DefineType("ScalarHelpers",
            TypeAttributes.Public | TypeAttributes.Abstract | TypeAttributes.Sealed);
        collections = new CollectionMethodTranslator(helpers);
    }

    /// <summary>Maps supported values, collections, and by-reference types to their device representations.</summary>
    /// <param name="type">The original value or reference type.</param>
    /// <returns>The mapped type; unsupported types are left unchanged for separate compatibility validation.</returns>
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

    /// <summary>Checks whether a signature uses supported values, recognized native views, or translated collections.</summary>
    /// <param name="type">The signature type, including void or by-reference types.</param>
    /// <returns>Whether the signature category is supported; native payload validation occurs during mapping.</returns>
    internal static bool IsCompatible(Type type) => type == typeof(void) ||
        DeviceValueType.IsSupported(type.IsByRef ? type.GetElementType()! : type) ||
        DeviceNativeTypes.IsSupported(type.IsByRef ? type.GetElementType()! : type) ||
        CapturedCollection.ElementType(type.IsByRef ? type.GetElementType()! : type) is not null;

    /// <summary>Emits a translated collection call, mapped helper, or unchanged compatible method call.</summary>
    /// <param name="il">The destination IL generator.</param>
    /// <param name="method">The original called method.</param>
    /// <param name="emitContext">Emits the invocation context when the replacement requires it.</param>
    /// <param name="receiverType">The original receiver type inferred from the stack, when available.</param>
    internal void EmitCall(ILGenerator il, MethodInfo method, Action<ILGenerator> emitContext,
        Type? receiverType = null)
    {
        if (collections.TryEmit(il, method, receiverType))
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
    /// <exception cref="NotSupportedException">The helper uses unsupported instructions or framework operations.</exception>
    internal MethodInfo Translate(MethodInfo method)
    {
        if (methods.TryGetValue(method, out var existing))
            return existing;
        if (method.Module.Assembly == typeof(ILGPU.Atomic).Assembly)
            return method;
        DeviceNumericMethods.Validate(method);
        if (DeviceNumericMethods.ExtendedMathReplacement(method) is { } implementation)
        {
            var replacement = MapType(method.ReturnType) == method.ReturnType
                ? implementation : TranslateMathResult(method, implementation);
            methods.Add(method, replacement);
            return replacement;
        }
        if (DeviceNumericMethods.IsLowered(method))
            return method.Name == nameof(Math.Clamp) ? TranslateClamp(method) : TranslateArithmeticMethod(method);
        DeviceInstructionValidation.ValidateCallGraph(method, validated);
        if (RequiresTranslation(method, new HashSet<MethodBase>()))
            return TranslateCore(method);
        methods.Add(method, method);
        return method;
    }

    /// <summary>Copies a native math tuple result into the caller's automatically mapped value layout.</summary>
    /// <remarks>SinCos and DivRem return framework tuples whose auto layout is replaced inside lambdas.
    /// The arithmetic implementation retains its native signature; this bridge preserves both result fields.</remarks>
    private MethodInfo TranslateMathResult(MethodInfo method, MethodInfo implementation)
    {
        AllowAccess(implementation.Module.Assembly);
        Type resultType = method.ReturnType;
        Type mappedResult = MapType(resultType);
        var parameters = method.GetParameters().Select(parameter => MapType(parameter.ParameterType))
            .Append(typeof(DeviceExecutionContext)).ToArray();
        var bridge = helpers.DefineMethod("MathResult" + methods.Count,
            MethodAttributes.Public | MethodAttributes.Static, mappedResult, parameters);
        var il = bridge.GetILGenerator();
        for (int i = 0; i < parameters.Length; i++)
            il.Emit(OpCodes.Ldarg, (short)i);
        il.Emit(OpCodes.Call, implementation);
        var native = il.DeclareLocal(resultType);
        var mapped = il.DeclareLocal(mappedResult);
        il.Emit(OpCodes.Stloc, native);
        il.Emit(OpCodes.Ldloca, mapped);
        il.Emit(OpCodes.Initobj, mappedResult);
        foreach (var field in resultType.GetFields(BindingFlags.Instance | BindingFlags.Public))
        {
            il.Emit(OpCodes.Ldloca, mapped);
            il.Emit(OpCodes.Ldloca, native);
            il.Emit(OpCodes.Ldfld, field);
            il.Emit(OpCodes.Stfld, DeviceValueType.MapField(field));
        }
        il.Emit(OpCodes.Ldloc, mapped);
        il.Emit(OpCodes.Ret);
        return bridge;
    }

    /// <summary>Emits a compatible value constructor, translating values and numeric operations when required.</summary>
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
        AllowAccess(method.Module.Assembly);
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
            if (stackTypes.TryGetValue(instruction.Offset, out var arithmeticStack) &&
                TryEmitArithmetic(il, instruction, arithmeticStack, emitContext))
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
                int receiverOffset = called.GetParameters().Length + 1;
                Type? receiverType = !called.IsStatic &&
                    stackTypes.TryGetValue(instruction.Offset, out var receiverStack) &&
                    receiverStack.Length >= receiverOffset ? receiverStack[^receiverOffset] : null;
                EmitCall(il, called, emitContext, receiverType);
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
                        AllowAccess(field.Module.Assembly);
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

    /// <summary>Allows generated host initializers to read private values from a transitive helper assembly.</summary>
    /// <param name="assembly">The assembly owning the translated method or its mapped static field.</param>
    private void AllowAccess(Assembly assembly)
    {
        string name = assembly.GetName().Name!;
        if (!accessAssemblies.Add(name))
            return;
        var attribute = typeof(System.Runtime.CompilerServices.IgnoresAccessChecksToAttribute)
            .GetConstructor([typeof(string)])!;
        generatedAssembly.SetCustomAttribute(new CustomAttributeBuilder(attribute, [name]));
    }

    /// <summary>Emits a nullable payload read that defers the absent-value exception to the host.</summary>
    private static void EmitNullableValue(ILGenerator il, MethodInfo method, Action<ILGenerator> emitContext)
    {
        var fields = method.DeclaringType!.GetFields(BindingFlags.Instance | BindingFlags.NonPublic);
        var value = fields.Single(field => field.Name == "value");
        // Nullable<Boolean> has two Boolean fields; the payload is not its presence flag.
        var present = fields.Single(field => field != value && field.FieldType == typeof(bool));
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
    /// <summary>Emits CLR sizeof constants, stack-compatible narrow conversions, guarded integer division/remainder, or floating remainder.</summary>
    /// <returns>True if a replacement was emitted; false for an instruction requiring no arithmetic lowering.</returns>
    internal static bool TryEmitArithmetic(ILGenerator il, LoopInstruction instruction,
        Type?[] stack, Action<ILGenerator> emitContext)
    {
        if (instruction.Code == OpCodes.Sizeof)
        {
            var type = (Type)instruction.Operand!;
            if (!DeviceValueType.IsSupported(type))
                throw new NotSupportedException($"sizeof requires a supported unmanaged GPU value type: '{type}'.");
            // Automatic mirrors can change field widths and packing. Preserve the source CLR size.
            il.Emit(OpCodes.Ldc_I4, ILGPU.Interop.SizeOf(type));
            return true;
        }
        if (DeviceArithmetic.IsNarrowingConversion(instruction.Code))
        {
            // CIL narrows the value but leaves an I4 on the evaluation stack. Keep the
            // backend phi inputs as Int32 instead of merging signed Int8/Int16 values.
            bool unsignedNarrow = instruction.Code == OpCodes.Conv_U1 || instruction.Code == OpCodes.Conv_U2;
            int bits = instruction.Code == OpCodes.Conv_I1 || instruction.Code == OpCodes.Conv_U1 ? 8 : 16;
            il.Emit(unsignedNarrow ? OpCodes.Conv_U4 : OpCodes.Conv_I4);
            if (unsignedNarrow)
            {
                il.Emit(OpCodes.Ldc_I4, (1 << bits) - 1);
                il.Emit(OpCodes.And);
            }
            else
            {
                il.Emit(OpCodes.Ldc_I4, 32 - bits);
                il.Emit(OpCodes.Shl);
                il.Emit(OpCodes.Ldc_I4, 32 - bits);
                il.Emit(OpCodes.Shr);
            }
            return true;
        }
        if (!DeviceArithmetic.IsDivision(instruction.Code))
            return false;
        int width = stack.Length < 2 ? -1 : DeviceArithmetic.IntegerWidth(stack[^2]);
        if (width == 0)
        {
            if (instruction.Code != OpCodes.Rem)
                return false;
            emitContext(il);
            il.Emit(OpCodes.Call, typeof(DeviceArithmetic).GetMethod(stack[^2] == typeof(float)
                ? nameof(DeviceArithmetic.RemainderSingle) : nameof(DeviceArithmetic.RemainderDouble))!);
            return true;
        }
        if (width < 0)
            throw new NotSupportedException("Cannot determine the integer width for checked device arithmetic.");
        bool unsigned = instruction.Code == OpCodes.Div_Un || instruction.Code == OpCodes.Rem_Un;
        bool remainder = instruction.Code == OpCodes.Rem || instruction.Code == OpCodes.Rem_Un;
        string name = (remainder ? "Remainder" : "Divide") + (unsigned ? "UInt" : "Int") + width;
        emitContext(il);
        il.Emit(OpCodes.Call, typeof(DeviceArithmetic).GetMethod(name)!);
        return true;
    }

    /// <summary>Binds supported Abs, Min, Max, and CopySign calls to their explicit device implementations.</summary>
    private MethodInfo TranslateArithmeticMethod(MethodInfo original)
    {
        var type = original.ReturnType;
        var parameters = original.GetParameters();
        var builder = helpers.DefineMethod(original.Name + methods.Count, MethodAttributes.Public | MethodAttributes.Static,
            type, parameters.Select(parameter => parameter.ParameterType).Append(typeof(DeviceExecutionContext)).ToArray());
        methods.Add(original, builder);
        var il = builder.GetILGenerator();
        string suffix = type == typeof(sbyte) ? "Int8" : type == typeof(short) ? "Int16" :
            type == typeof(int) ? "Int32" : type == typeof(long) ? "Int64" :
            type == typeof(float) ? "Single" : "Double";
        for (int i = 0; i <= parameters.Length; i++)
            il.Emit(OpCodes.Ldarg, (short)i);
        il.Emit(OpCodes.Call, typeof(DeviceArithmetic).GetMethod(original.Name + suffix)!);
        il.Emit(OpCodes.Ret);
        return builder;
    }
    /// <summary>Emits ordered Clamp comparisons and a deferred error for inverted bounds.</summary>
    private MethodInfo TranslateClamp(MethodInfo original)
    {
        var type = original.ReturnType;
        var builder = helpers.DefineMethod("Clamp" + methods.Count, MethodAttributes.Public | MethodAttributes.Static,
            type, [type, type, type, typeof(DeviceExecutionContext)]);
        methods.Add(original, builder);
        var il = builder.GetILGenerator();
        var valid = il.DefineLabel();
        var minimum = il.DefineLabel();
        var maximum = il.DefineLabel();
        bool unsigned = type == typeof(byte) || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong);
        // Use ordered comparisons for floating-point NaN, matching Math.Clamp.
        il.Emit(OpCodes.Ldarg_1);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(unsigned ? OpCodes.Cgt_Un : OpCodes.Cgt);
        il.Emit(OpCodes.Brfalse, valid);
        il.Emit(OpCodes.Ldarga_S, (byte)3);
        il.Emit(OpCodes.Ldc_I4, 10);
        il.Emit(OpCodes.Call, typeof(DeviceExecutionContext).GetMethod(nameof(DeviceExecutionContext.Fail))!);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ret);
        il.MarkLabel(valid);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldarg_1);
        il.Emit(unsigned ? OpCodes.Blt_Un : OpCodes.Blt, minimum);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(unsigned ? OpCodes.Bgt_Un : OpCodes.Bgt, maximum);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ret);
        il.MarkLabel(minimum);
        il.Emit(OpCodes.Ldarg_1);
        il.Emit(OpCodes.Ret);
        il.MarkLabel(maximum);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(OpCodes.Ret);
        return builder;
    }

    /// <summary>Checks whether a signature, local, field, transitive call, or numeric operation needs translation.</summary>
    /// <param name="method">The helper to inspect.</param>
    /// <param name="visited">The methods already inspected along the current call graph.</param>
    /// <returns>True when value mapping, arithmetic lowering, or translation of a transitive call is required.</returns>
    private static bool RequiresTranslation(MethodBase method, HashSet<MethodBase> visited)
    {
        if (method.Module.Assembly == typeof(ILGPU.Atomic).Assembly || !visited.Add(method))
            return false;
        if (DeviceNumericMethods.IsMath(method))
        {
            DeviceNumericMethods.Validate(method);
            return DeviceNumericMethods.IsLowered(method);
        }
        if ((!method.IsStatic && MapType(method.DeclaringType!) != method.DeclaringType) ||
            (method is MethodInfo info && MapType(info.ReturnType) != info.ReturnType) ||
            method.GetParameters().Any(parameter => MapType(parameter.ParameterType) != parameter.ParameterType))
            return true;

        var body = method.GetMethodBody();
        if (body is null)
            return false;
        if (body.LocalVariables.Any(local => MapType(local.LocalType) != local.LocalType))
            return true;
        if (body.ExceptionHandlingClauses.Count == 0)
        {
            var instructions = LoopIL.Read(method);
            if (instructions.Any(instruction => instruction.Code == OpCodes.Sizeof ||
                DeviceArithmetic.IsNarrowingConversion(instruction.Code)))
                return true;
            if (instructions.Any(instruction => DeviceArithmetic.IsDivision(instruction.Code)))
            {
                var stacks = LoopStackTypes.Analyze(method, instructions);
                if (instructions.Any(instruction => DeviceArithmetic.IsDivision(instruction.Code) &&
                    stacks.TryGetValue(instruction.Offset, out var stack) && stack.Length >= 2 &&
                    (DeviceArithmetic.IntegerWidth(stack[^2]) != 0 || instruction.Code == OpCodes.Rem)))
                    return true;
            }
        }
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
