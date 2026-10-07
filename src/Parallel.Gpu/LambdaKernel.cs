using System.Reflection;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.Runtime;

namespace ParallelGpu;

internal sealed class LambdaKernel
{
    private sealed record Capture(FieldInfo[] Path, FieldInfo Destination, Type? Element)
    {
        internal object Read(object? target)
        {
            object? value = target;
            foreach (var field in Path)
                value = field.GetValue(field.IsStatic ? null : value ??
                    throw new ArgumentException("A captured object is null.", "body"));
            return value ?? throw new ArgumentException(
                $"Captured collection '{Path[^1].Name}' is null.", "body");
        }
    }

    private readonly Type stateType;
    private readonly Capture[] captures;
    private readonly Action<int, int, object> launch;

    private LambdaKernel(Type stateType, Capture[] captures, Action<int, int, object> launch)
    {
        this.stateType = stateType;
        this.captures = captures;
        this.launch = launch;
    }

    internal static void ValidateElementType(Type type)
    {
        if (!CapturedCollection.IsNumeric(type))
            throw new NotSupportedException(
                $"'{type.Name}' is not a supported GPU loop element type. " +
                "Use float, int, uint, long, ulong, double, short, ushort, byte or sbyte.");
    }

    internal void Execute(GpuParallel gpu, Action<int> body, int start, int count, Type? expectedElement)
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
                values[i] is CapturedCollection collection ? collection.View : values[i]);

        launch(count, start, state);
        gpu.Synchronize();
        transfers.CopyBack();
    }

    internal static LambdaKernel Compile(Accelerator accelerator, MethodInfo source)
    {
        var instructions = LoopIL.Read(source);
        var sourceBody = source.GetMethodBody()!;
        var assembly = AssemblyBuilder.DefineDynamicAssembly(
            new AssemblyName("ParallelGpu.Loop." + Guid.NewGuid().ToString("N")),
            AssemblyBuilderAccess.RunAndCollect);
        var accessAttribute = typeof(System.Runtime.CompilerServices.IgnoresAccessChecksToAttribute)
            .GetConstructor([typeof(string)])!;
        var referencedAssemblies = instructions.Select(instruction => instruction.Operand)
            .OfType<MemberInfo>().Select(member => member.Module.Assembly)
            .Append(source.Module.Assembly).Append(typeof(DeviceCollection).Assembly).Distinct();
        foreach (var referencedAssembly in referencedAssemblies)
            assembly.SetCustomAttribute(new CustomAttributeBuilder(accessAttribute,
                [referencedAssembly.GetName().Name!]));
        var module = assembly.DefineDynamicModule("Loop");
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
                while (fields.Count == 0 || !IsCaptureType(fields[^1].FieldType))
                {
                    if (++i >= instructions.Count || instructions[i].Code != OpCodes.Ldfld ||
                        branches.Contains(instructions[i].Offset))
                        throw Unsupported("Managed objects and mutation of captured scalar variables");
                    il.MarkLabel(labels[instructions[i].Offset]);
                    fields.Add((FieldInfo)instructions[i].Operand!);
                }
                string key = string.Join("/", fields.Select(field =>
                    field.Module.ModuleVersionId + ":" + field.MetadataToken));
                if (!paths.TryGetValue(key, out var capture))
                {
                    var type = fields[^1].FieldType;
                    capture = (fields.ToArray(), state.DefineField("Capture" + paths.Count,
                        MapType(type), FieldAttributes.Public), CapturedCollection.ElementType(type));
                    paths.Add(key, capture);
                }
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldfld, capture.Field);
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
                EmitCall(il, (MethodInfo)instruction.Operand!);
                continue;
            }
            if (instruction.Code == OpCodes.Readonly)
                continue; // The replacement for ldelema is an ordinary ref-returning method.
            if (instruction.Code == OpCodes.Newarr || instruction.Code == OpCodes.Newobj ||
                instruction.Code == OpCodes.Box || instruction.Code == OpCodes.Unbox ||
                instruction.Code == OpCodes.Unbox_Any || instruction.Code == OpCodes.Castclass ||
                instruction.Code == OpCodes.Isinst || instruction.Code == OpCodes.Throw ||
                instruction.Code == OpCodes.Rethrow || instruction.Code == OpCodes.Stfld ||
                instruction.Code == OpCodes.Stsfld || instruction.Code == OpCodes.Ldfld ||
                instruction.Code == OpCodes.Ldflda || instruction.Code == OpCodes.Ldsflda ||
                instruction.Code == OpCodes.Ldnull || instruction.Code == OpCodes.Ldftn ||
                instruction.Code == OpCodes.Ldvirtftn || instruction.Code == OpCodes.Ldtoken ||
                instruction.Code == OpCodes.Calli || instruction.Code == OpCodes.Constrained)
                throw Unsupported($"Instruction '{instruction.Code.Name}'");
            LoopIL.Emit(il, instruction, labels);
        }

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
        var stateType = state.CreateType()!;
        var captures = paths.Values.Select(capture => new Capture(capture.Path,
            stateType.GetField(capture.Field.Name)!, capture.Element)).ToArray();
        try
        {
            var loader = typeof(LambdaKernel).GetMethod(nameof(Load), BindingFlags.NonPublic | BindingFlags.Static)!
                .MakeGenericMethod(stateType);
            var launch = (Action<int, int, object>)loader.Invoke(null,
                [accelerator, stateType.GetMethod("Execute")!])!;
            return new(stateType, captures, launch);
        }
        catch (TargetInvocationException error) when (error.InnerException is not null)
        {
            throw new NotSupportedException(
                $"Cannot compile GPU loop body '{source.Name}'. Use supported numeric operations, " +
                "array/list indexing and device-compatible scalar methods.", error.InnerException);
        }
    }

    private static Action<int, int, object> Load<TState>(Accelerator accelerator, MethodInfo entry)
        where TState : struct
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, TState>(
            (Action<Index1D, int, TState>)entry.CreateDelegate(typeof(Action<Index1D, int, TState>)));
        return (count, start, state) => kernel(count, start, (TState)state);
    }

    private static bool IsCaptureType(Type type) => IsScalar(type) ||
        CapturedCollection.ElementType(type) is not null;

    private static bool IsScalar(Type type) => CapturedCollection.IsNumeric(type) ||
        type == typeof(bool) || type == typeof(char) || type.IsEnum;

    private static Type MapType(Type type)
    {
        if (CapturedCollection.ElementType(type) is not null)
            return typeof(DeviceCollection);
        if (IsScalar(type) || (type.IsByRef && IsScalar(type.GetElementType()!)))
            return type;
        throw Unsupported($"Type '{type}'");
    }

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
        il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(name)!.MakeGenericMethod(type));
        return true;
    }

    private static void EmitCall(ILGenerator il, MethodInfo method)
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
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(name)!.MakeGenericMethod(matrixElement));
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
                    replacement = replacement.MakeGenericMethod(element);
                il.Emit(OpCodes.Call, replacement);
                return;
            }
        }
        if ((!method.IsStatic && !IsScalar(declaringType)) ||
            method.GetParameters().Any(parameter => !IsScalar(
                parameter.ParameterType.IsByRef ? parameter.ParameterType.GetElementType()! : parameter.ParameterType)) ||
            (method.ReturnType != typeof(void) && !IsScalar(method.ReturnType)))
            throw Unsupported($"Method '{declaringType.FullName}.{method.Name}'");
        il.Emit(OpCodes.Call, method);
    }

    private static NotSupportedException Unsupported(string feature) => new(
        feature + " is not supported inside GPU loops. Use numeric locals and captured arrays or List<T>; " +
        "write results by index. No CPU fallback is performed.");
}




