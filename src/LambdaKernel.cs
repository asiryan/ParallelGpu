using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.Runtime;

namespace ParallelGpu;

/// <summary>Compiles an ordinary loop delegate into a cached ILGPU kernel with translated captures.</summary>
/// <remarks>Compiled instructions are reused, while closure values and collection transfers are
/// rebuilt for every invocation.</remarks>
internal sealed class LambdaKernel
{
    /// <summary>Persistent translated bodies shared by executors; no closures or accelerator resources are retained.</summary>
    private static readonly ConcurrentDictionary<MethodInfo, Lazy<TranslatedBody>> translatedBodies = new();
    /// <summary>The generated entry point, state layout, and capture paths for one ordinary method.</summary>
    /// <param name="StateType">The public generated kernel state.</param>
    /// <param name="Captures">The host capture paths and their generated fields.</param>
    /// <param name="Entry">The generated ILGPU entry point.</param>
    private sealed record TranslatedBody(Type StateType, LoopCapture[] Captures, MethodInfo Entry);

    /// <summary>The generated value type containing device views and captured scalar values.</summary>
    private readonly Type stateType;
    /// <summary>The field mappings used to read current closure values on every invocation.</summary>
    private readonly LoopCapture[] captures;
    /// <summary>The accelerator-bound launcher accepting an iteration count, lower bound, and boxed state.</summary>
    private readonly Action<int, int, object> launch;

    /// <summary>Stores the generated state type, capture mappings, and bound kernel launcher.</summary>
    /// <param name="stateType">The generated value type used for kernel arguments.</param>
    /// <param name="captures">Mappings from host fields to state fields.</param>
    /// <param name="launch">The bound device launcher.</param>
    private LambdaKernel(Type stateType, LoopCapture[] captures, Action<int, int, object> launch)
    {
        this.stateType = stateType;
        this.captures = captures;
        this.launch = launch;
    }

    /// <summary>Rejects collection element types containing unsupported fields or managed references.</summary>
    /// <param name="type">The requested collection element type.</param>
    /// <exception cref="NotSupportedException">The element type is unsupported.</exception>
    internal static void ValidateElementType(Type type)
    {
        if (!CapturedCollection.IsSupportedElementType(type))
            throw new NotSupportedException(
                $"'{type.Name}' is not a supported GPU loop element type. " +
                "Use primitives, enums, or device-compatible structures without managed references.");
    }

    /// <summary>Uploads current captures, executes the cached kernel, and copies written collections back.</summary>
    /// <param name="gpu">The executor that owns this kernel's accelerator.</param>
    /// <param name="body">The current delegate instance supplying closure values.</param>
    /// <param name="start">The inclusive lower bound of the loop range.</param>
    /// <param name="count">The positive number of iterations to launch.</param>
    /// <param name="expectedElement">The required collection element type, or null for mixed types.</param>
    /// <exception cref="ArgumentException">A capture is null or has an unexpected collection element type.</exception>
    /// <exception cref="NotSupportedException">A captured runtime collection is unsupported.</exception>
    internal void Execute(Gpu gpu, Action<int> body, int start, int count, Type? expectedElement)
    {
        var state = Activator.CreateInstance(stateType)!;
        using var transfers = new CollectionTransfers(gpu);

        // Validate every capture before allocating memory or launching the kernel.
        var values = captures.Select(capture => capture.Read(body.Target)).ToArray();
        for (int i = 0; i < captures.Length; i++)
        {
            var capture = captures[i];
            if (capture.Element is not null && expectedElement is not null &&
                capture.Element != expectedElement)
                throw new ArgumentException(
                    $"Captured collection '{capture.Path[^1].Name}' contains {capture.Element.Name}, " +
                    $"but For<{expectedElement.Name}> requires that element type. " +
                    "Use the non-generic For overload for mixed collection types.", nameof(body));
            if (capture.Element is not null && values[i] is not Array &&
                (!values[i].GetType().IsGenericType ||
                 values[i].GetType().GetGenericTypeDefinition() != typeof(List<>)))
                throw new NotSupportedException(
                    "Captured IList<T>/IReadOnlyList<T> must refer to an array or List<T>.");
        }
        for (int i = 0; i < captures.Length; i++)
            if (captures[i].Element is Type element)
                values[i] = transfers.Add(values[i], element);
        transfers.Upload();
        for (int i = 0; i < captures.Length; i++)
            captures[i].Destination.SetValue(state,
                values[i] switch
                {
                    CapturedCollection collection => collection.View,
                    bool value => value ? 1 : 0,
                    char value => (int)value,
                    _ => DeviceValueType.ToDevice(values[i], captures[i].Path[^1].FieldType)
                });

        launch(count, start, state);
        gpu.Synchronize();
        transfers.CopyBack();
    }

    /// <summary>Loads a shared translated loop body on an accelerator without retaining closure instances.</summary>
    /// <param name="accelerator">The accelerator used to compile and launch the kernel.</param>
    /// <param name="source">The original delegate method to translate.</param>
    /// <returns>A compiled kernel with capture metadata and a bound launcher.</returns>
    /// <exception cref="NotSupportedException">The body cannot be translated or compiled for the device.</exception>
    internal static LambdaKernel Compile(Accelerator accelerator, MethodInfo source)
    {
        var translated = translatedBodies.GetOrAdd(source,
            method => new Lazy<TranslatedBody>(() => TranslateBody(method))).Value;
        try
        {
            var loader = typeof(LambdaKernel).GetMethod(nameof(Load), BindingFlags.NonPublic | BindingFlags.Static)!
                .MakeGenericMethod(translated.StateType);
            var launch = (Action<int, int, object>)loader.Invoke(null, [accelerator, translated.Entry])!;
            return new(translated.StateType, translated.Captures, launch);
        }
        catch (TargetInvocationException error) when (error.InnerException is not null)
        {
            throw new NotSupportedException(
                $"Cannot compile GPU loop body '{source.Name}'. Use device-compatible value operations, " +
                "array/list indexing and device-compatible methods.", error.InnerException);
        }
    }

    /// <summary>Rewrites a managed loop into a persistent template shared across accelerators.</summary>
    /// <param name="source">The original ordinary loop method.</param>
    /// <returns>The generated state, capture paths, and entry point.</returns>
    /// <remarks>ILGPU CPU code can outlive an executor during finalization. Keeping the source assembly
    /// noncollectible avoids invalid dynamic-method handles; one template is generated per body method.</remarks>
    private static TranslatedBody TranslateBody(MethodInfo source)
    {
        var instructions = LoopIL.Read(source);
        var sourceBody = source.GetMethodBody()!;
        var assembly = AssemblyBuilder.DefineDynamicAssembly(
            new AssemblyName("ParallelGpu.Loop." + Guid.NewGuid().ToString("N")),
            AssemblyBuilderAccess.Run);
        var accessAttribute = typeof(System.Runtime.CompilerServices.IgnoresAccessChecksToAttribute)
            .GetConstructor([typeof(string)])!;
        // Generated code must call internal collection helpers and private scalar helpers from the original body.
        var referencedAssemblies = instructions.Select(instruction => instruction.Operand)
            .OfType<MemberInfo>().Select(member => member.Module.Assembly)
            .Append(source.Module.Assembly).Append(typeof(DeviceCollection).Assembly).Distinct();
        foreach (var referencedAssembly in referencedAssemblies)
            assembly.SetCustomAttribute(new CustomAttributeBuilder(accessAttribute,
                [referencedAssembly.GetName().Name!]));
        var module = assembly.DefineDynamicModule("Loop");
        var scalarMethods = new ScalarMethodTranslator(module);
        var state = module.DefineType("LoopState", TypeAttributes.Public | TypeAttributes.Sealed |
            TypeAttributes.SequentialLayout, typeof(ValueType));
        var body = state.DefineMethod("Body", MethodAttributes.Public | MethodAttributes.Static,
            typeof(void), [state.MakeByRefType(), typeof(int)]);
        var il = body.GetILGenerator();
        foreach (var local in sourceBody.LocalVariables)
        {
            if (local.IsPinned)
                throw Unsupported("Pinned variables");
            il.DeclareLocal(MapType(local.LocalType));
        }
        // Keep labels keyed by original offsets because rewritten instructions can have different sizes.
        var labels = instructions.ToDictionary(instruction => instruction.Offset, _ => il.DefineLabel());
        var branches = instructions.Where(instruction =>
                instruction.Code.OperandType is OperandType.InlineBrTarget or OperandType.ShortInlineBrTarget)
            .Select(instruction => (int)instruction.Operand!)
            .Concat(instructions.Where(instruction => instruction.Code == OpCodes.Switch)
                .SelectMany(instruction => (int[])instruction.Operand!)).ToHashSet();
        var paths = new Dictionary<string, (FieldInfo[] Path, FieldBuilder Field, Type? Element)>();

        for (int i = 0; i < instructions.Count; i++)
        {
            var instruction = instructions[i];
            il.MarkLabel(labels[instruction.Offset]);
            var argument = LoopIL.Argument(instruction);
            if ((!source.IsStatic && argument == 0) || instruction.Code == OpCodes.Ldsfld)
            {
                var fields = new List<FieldInfo>();
                if (instruction.Code == OpCodes.Ldsfld)
                    fields.Add((FieldInfo)instruction.Operand!);
                // Collapse a closure field chain only when no branch can enter it partway through.
                while (fields.Count == 0 || !IsCaptureType(fields[^1].FieldType))
                {
                    if (++i >= instructions.Count ||
                        (instructions[i].Code != OpCodes.Ldfld && instructions[i].Code != OpCodes.Ldflda) ||
                        branches.Contains(instructions[i].Offset))
                        throw Unsupported("Managed objects and mutation of captured scalar variables");
                    il.MarkLabel(labels[instructions[i].Offset]);
                    fields.Add((FieldInfo)instructions[i].Operand!);
                }
                // Closed declaring types distinguish fields sharing a metadata token across generic instantiations.
                string key = string.Join("/", fields.Select(field =>
                    field.Module.ModuleVersionId + ":" + field.MetadataToken + ":" +
                    field.DeclaringType?.AssemblyQualifiedName));
                if (!paths.TryGetValue(key, out var capture))
                {
                    var type = fields[^1].FieldType;
                    capture = (fields.ToArray(), state.DefineField("Capture" + paths.Count,
                        MapCaptureType(type), FieldAttributes.Public), CapturedCollection.ElementType(type));
                    paths.Add(key, capture);
                }
                bool address = instructions[i].Code == OpCodes.Ldflda;
                if (address)
                    ReadOnlyCapture.Validate(instructions, i);
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldfld, capture.Field);
                if (address)
                {
                    // Reading an instance member needs an address; use a per-iteration value copy.
                    var local = il.DeclareLocal(ScalarMethodTranslator.MapType(fields[^1].FieldType));
                    il.Emit(OpCodes.Stloc, local);
                    il.Emit(OpCodes.Ldloca, local);
                }
                continue;
            }
            if (argument is not null)
            {
                if (argument != (source.IsStatic ? 0 : 1))
                    throw Unsupported("Additional loop arguments");
                il.Emit(OpCodes.Ldarg_1);
                continue;
            }
            if (instruction.Code == OpCodes.Starg || instruction.Code == OpCodes.Starg_S ||
                instruction.Code == OpCodes.Ldarga || instruction.Code == OpCodes.Ldarga_S)
            {
                int expected = source.IsStatic ? 0 : 1;
                if ((int)instruction.Operand! != expected)
                    throw Unsupported("Mutation of captured state");
                il.Emit(instruction.Code == OpCodes.Starg || instruction.Code == OpCodes.Starg_S
                    ? OpCodes.Starg : OpCodes.Ldarga, (short)1);
                continue;
            }
            if (TryArrayOperation(il, instruction))
                continue;
            if (instruction.Code == OpCodes.Call || instruction.Code == OpCodes.Callvirt)
            {
                if (instruction.Operand is ConstructorInfo constructor)
                    scalarMethods.EmitConstructor(il, instruction.Code, constructor);
                else
                    EmitCall(il, (MethodInfo)instruction.Operand!, scalarMethods);
                continue;
            }
            if (instruction.Code == OpCodes.Newobj)
            {
                scalarMethods.EmitConstructor(il, instruction.Code, (ConstructorInfo)instruction.Operand!);
                continue;
            }
            if (instruction.Code == OpCodes.Ldfld || instruction.Code == OpCodes.Ldflda ||
                instruction.Code == OpCodes.Stfld)
            {
                var field = (FieldInfo)instruction.Operand!;
                if (!IsScalar(field.DeclaringType!) || !IsScalar(field.FieldType))
                    throw Unsupported($"Field '{field.Name}'");
                il.Emit(instruction.Code, DeviceValueType.MapField(field));
                continue;
            }
            if (instruction.Code == OpCodes.Readonly)
                continue; // The replacement for ldelema is an ordinary ref-returning method.
            if (instruction.Code == OpCodes.Newarr ||
                instruction.Code == OpCodes.Box || instruction.Code == OpCodes.Unbox ||
                instruction.Code == OpCodes.Unbox_Any || instruction.Code == OpCodes.Castclass ||
                instruction.Code == OpCodes.Isinst || instruction.Code == OpCodes.Throw ||
                instruction.Code == OpCodes.Rethrow ||
                instruction.Code == OpCodes.Stsfld || instruction.Code == OpCodes.Ldsflda ||
                instruction.Code == OpCodes.Ldnull || instruction.Code == OpCodes.Ldftn ||
                instruction.Code == OpCodes.Ldvirtftn || instruction.Code == OpCodes.Ldtoken ||
                instruction.Code == OpCodes.Calli || instruction.Code == OpCodes.Constrained)
                throw Unsupported($"Instruction '{instruction.Code.Name}'");
            LoopIL.Emit(il, instruction.Operand is Type operandType
                ? instruction with { Operand = ScalarMethodTranslator.MapType(operandType) } : instruction, labels);
        }

        // The ILGPU entry point converts a zero-based device index into the original loop index.
        var entry = state.DefineMethod("Execute", MethodAttributes.Public | MethodAttributes.Static,
            typeof(void), [typeof(Index1D), typeof(int), state]);
        var entryIL = entry.GetILGenerator();
        entryIL.Emit(OpCodes.Ldarga_S, (byte)2);
        entryIL.Emit(OpCodes.Ldarg_1);
        entryIL.Emit(OpCodes.Ldarga_S, (byte)0);
        entryIL.Emit(OpCodes.Call, typeof(Index1D).GetProperty(nameof(Index1D.X))!.GetMethod!);
        entryIL.Emit(OpCodes.Add);
        entryIL.Emit(OpCodes.Call, body);
        entryIL.Emit(OpCodes.Ret);
        scalarMethods.Complete();
        var stateType = state.CreateType()!;
        var captures = paths.Values.Select(capture => new LoopCapture(capture.Path,
            stateType.GetField(capture.Field.Name)!, capture.Element)).ToArray();
        return new(stateType, captures, stateType.GetMethod("Execute")!);
    }

    /// <summary>Binds the generated entry point to a strongly typed ILGPU launcher.</summary>
    /// <typeparam name="TState">The generated kernel state value type.</typeparam>
    /// <param name="accelerator">The accelerator that loads the kernel.</param>
    /// <param name="entry">The generated static kernel entry method.</param>
    /// <returns>A launcher accepting the loop count, lower bound, and boxed state.</returns>
    private static Action<int, int, object> Load<TState>(Accelerator accelerator, MethodInfo entry)
        where TState : struct
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, TState>(
            (Action<Index1D, int, TState>)entry.CreateDelegate(typeof(Action<Index1D, int, TState>)));
        return (count, start, state) => kernel(count, start, (TState)state);
    }

    /// <summary>Checks whether a captured field can be translated as a scalar or collection.</summary>
    /// <param name="type">The captured field type.</param>
    /// <returns>True when the field is a supported capture.</returns>
    private static bool IsCaptureType(Type type) => IsScalar(type) ||
        CapturedCollection.ElementType(type) is not null;

    /// <summary>Checks whether a type can be used as a numeric or simple scalar value.</summary>
    /// <param name="type">The candidate scalar type.</param>
    /// <returns>True for supported primitives, enums, and structures without managed references.</returns>
    private static bool IsScalar(Type type) => CapturedCollection.IsSupportedElementType(type);

    /// <summary>Maps captured Boolean and character scalars to blittable kernel argument fields.</summary>
    /// <param name="type">The captured scalar or collection type.</param>
    /// <returns>A 32-bit scalar field or the ordinary device representation.</returns>
    private static Type MapCaptureType(Type type) => type == typeof(bool) || type == typeof(char)
        ? typeof(int) : MapType(type);

    /// <summary>Maps managed collections to device views and character scalars to full-width UInt16 values.</summary>
    /// <param name="type">The captured field or local variable type.</param>
    /// <returns>The corresponding generated kernel type.</returns>
    private static Type MapType(Type type)
    {
        type = ScalarMethodTranslator.MapType(type);
        if (CapturedCollection.ElementType(type) is not null)
            return typeof(DeviceCollection);
        if (IsScalar(type) || (type.IsByRef && IsScalar(type.GetElementType()!)))
            return type;
        throw Unsupported($"Type '{type}'");
    }

    /// <summary>Maps element values to full-width characters or compatible device structure mirrors.</summary>
    /// <param name="type">The declared collection element type.</param>
    /// <returns>The device representation of a character, structure, or unchanged native value.</returns>
    private static Type MapElementValueType(Type type) => DeviceValueType.MapType(type);

    /// <summary>Rewrites supported array instructions as device collection operations.</summary>
    /// <param name="il">The destination IL generator.</param>
    /// <param name="instruction">The decoded array instruction to inspect.</param>
    /// <returns>True if the instruction was recognized and emitted; otherwise, false.</returns>
    private static bool TryArrayOperation(ILGenerator il, LoopInstruction instruction)
    {
        var code = instruction.Code;
        if (code == OpCodes.Ldelem_Ref)
        {
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.Row))!);
            return true;
        }
        if (code == OpCodes.Stelem_Ref)
            throw Unsupported("Replacing rows in a jagged array");
        if (code == OpCodes.Ldlen)
        {
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.Count))!);
            // Collection lengths are Int32, matching the supported loop extent.
            return true;
        }
        var type = code.Value switch
        {
            var v when v == OpCodes.Ldelem_I1.Value => typeof(sbyte),
            var v when v == OpCodes.Ldelem_U1.Value => typeof(byte),
            var v when v == OpCodes.Ldelem_I2.Value => typeof(short),
            var v when v == OpCodes.Ldelem_U2.Value => typeof(ushort),
            var v when v == OpCodes.Ldelem_I4.Value => typeof(int),
            var v when v == OpCodes.Ldelem_U4.Value => typeof(uint),
            var v when v == OpCodes.Ldelem_I8.Value => typeof(long),
            var v when v == OpCodes.Ldelem_R4.Value => typeof(float),
            var v when v == OpCodes.Ldelem_R8.Value => typeof(double),
            var v when v == OpCodes.Stelem_I1.Value => typeof(sbyte),
            var v when v == OpCodes.Stelem_I2.Value => typeof(short),
            var v when v == OpCodes.Stelem_I4.Value => typeof(int),
            var v when v == OpCodes.Stelem_I8.Value => typeof(long),
            var v when v == OpCodes.Stelem_R4.Value => typeof(float),
            var v when v == OpCodes.Stelem_R8.Value => typeof(double),
            var v when v == OpCodes.Ldelem.Value || v == OpCodes.Stelem.Value ||
                       v == OpCodes.Ldelema.Value => (Type)instruction.Operand!,
            _ => null
        };
        if (type is null)
            return false;
        ValidateElementType(type);
        string name = code == OpCodes.Ldelema ? nameof(DeviceCollection.Address) :
            code.Name!.StartsWith("stelem", StringComparison.Ordinal) ? nameof(DeviceCollection.Write) :
            nameof(DeviceCollection.Read);
        il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(name)!.MakeGenericMethod(MapElementValueType(type)));
        return true;
    }

    /// <summary>Rewrites collection calls and preserves supported device-compatible scalar calls.</summary>
    /// <param name="il">The destination IL generator.</param>
    /// <param name="method">The source method referenced by the call instruction.</param>
    /// <param name="scalarMethods">The translator preserving character values in scalar helper calls.</param>
    private static void EmitCall(ILGenerator il, MethodInfo method, ScalarMethodTranslator scalarMethods)
    {
        var declaringType = method.DeclaringType!;
        if (declaringType.IsArray && declaringType.GetArrayRank() == 2 &&
            CapturedCollection.ElementType(declaringType) is Type matrixElement)
        {
            string name = method.Name switch
            {
                "Get" => nameof(DeviceCollection.Read2D),
                "Set" => nameof(DeviceCollection.Write2D),
                "Address" => nameof(DeviceCollection.Address2D),
                _ => throw Unsupported($"Array method '{method.Name}'")
            };
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(name)!.MakeGenericMethod(MapElementValueType(matrixElement)));
            return;
        }
        if (declaringType == typeof(Array))
        {
            string name = method.Name switch
            {
                "get_Length" => nameof(DeviceCollection.Count),
                "get_LongLength" => nameof(DeviceCollection.LongCount),
                "GetLength" => nameof(DeviceCollection.GetLength),
                "GetLongLength" => nameof(DeviceCollection.GetLongLength),
                _ => throw Unsupported($"Array method '{method.Name}'")
            };
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(name)!);
            return;
        }
        if (declaringType.IsGenericType)
        {
            var definition = declaringType.GetGenericTypeDefinition();
            if (definition == typeof(List<>) || definition == typeof(IList<>) ||
                definition == typeof(IReadOnlyList<>) || definition == typeof(ICollection<>) ||
                definition == typeof(IReadOnlyCollection<>))
            {
                var element = declaringType.GetGenericArguments()[0];
                ValidateElementType(element);
                string name = method.Name switch
                {
                    "get_Item" => nameof(DeviceCollection.Read),
                    "set_Item" => nameof(DeviceCollection.Write),
                    "get_Count" => nameof(DeviceCollection.Count),
                    _ => throw Unsupported($"Collection method '{method.Name}'")
                };
                var replacement = typeof(DeviceCollection).GetMethod(name)!;
                if (replacement.IsGenericMethodDefinition)
                    replacement = replacement.MakeGenericMethod(MapElementValueType(element));
                il.Emit(OpCodes.Call, replacement);
                return;
            }
        }
        // Collection instance calls were replaced above; remaining helpers must use device-value signatures.
        if ((!method.IsStatic && !IsScalar(declaringType)) ||
            method.GetParameters().Any(parameter => !IsScalar(
                parameter.ParameterType.IsByRef ? parameter.ParameterType.GetElementType()! : parameter.ParameterType)) ||
            (method.ReturnType != typeof(void) && !IsScalar(method.ReturnType.IsByRef
                ? method.ReturnType.GetElementType()! : method.ReturnType)))
            throw Unsupported($"Method '{declaringType.FullName}.{method.Name}'");
        il.Emit(OpCodes.Call, scalarMethods.Translate(method));
    }

    /// <summary>Creates an exception identifying a feature unavailable inside GPU loop bodies.</summary>
    /// <param name="feature">The unsupported feature or operation.</param>
    /// <returns>An exception describing the unsupported feature and loop restrictions.</returns>
    private static NotSupportedException Unsupported(string feature) => new(
        feature + " is not supported inside GPU loops. Use device-compatible value locals and captured arrays or List<T>; " +
        "write results by index. No CPU fallback is performed.");
}
