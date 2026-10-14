using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Tracks IL stack types for collection reference comparisons and guarded integer arithmetic.</summary>
internal static class LoopStackTypes
{
    /// <summary>Marks a join of collection references without a shared supported host type.</summary>
    private sealed class MergedCollectionReference;
    /// <summary>Retains a common leaf type when unlike collection references meet at a control-flow join.</summary>
    private sealed class MergedCollectionReference<T>;

    /// <summary>Propagates original evaluation-stack types across reachable instructions and control-flow joins.</summary>
    /// <param name="method">The method supplying argument and local types.</param>
    /// <param name="instructions">Decoded instructions in source order.</param>
    /// <returns>Stack types before each reachable instruction, keyed by its original IL offset.</returns>
    internal static Dictionary<int, Type?[]> Analyze(MethodBase method, List<LoopInstruction> instructions)
    {
        var states = new Dictionary<int, Type?[]>();
        if (instructions.Count == 0)
            return states;
        var offsets = instructions.Select((instruction, index) => (instruction.Offset, index))
            .ToDictionary(item => item.Offset, item => item.index);
        var pending = new Queue<int>();
        states[instructions[0].Offset] = [];
        pending.Enqueue(0);
        while (pending.Count > 0)
        {
            int index = pending.Dequeue();
            var instruction = instructions[index];
            var stack = states[instruction.Offset].ToList();
            ValidateCollectionConversion(method, instruction, stack);
            Apply(method, instruction, stack);
            var code = instruction.Code;
            if (code.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch)
            {
                foreach (int target in instruction.Operand is int[] choices ? choices : [(int)instruction.Operand!])
                    Merge(offsets[target], stack);
            }
            if (code.FlowControl is not (FlowControl.Branch or FlowControl.Return or FlowControl.Throw) && index + 1 < instructions.Count)
                Merge(index + 1, stack);
        }
        return states;

        void Merge(int index, List<Type?> stack)
        {
            int offset = instructions[index].Offset;
            if (!states.TryGetValue(offset, out var previous))
            {
                states[offset] = stack.ToArray();
                pending.Enqueue(index);
                return;
            }
            if (previous.Length != stack.Count)
                throw new NotSupportedException("The GPU loop contains incompatible IL stack shapes.");
            bool changed = false;
            for (int i = 0; i < previous.Length; i++)
            {
                var merged = previous[i] == stack[i] ? previous[i] :
                    previous[i] == typeof(void) ? stack[i] : stack[i] == typeof(void) ? previous[i] : MergeTypes(previous[i], stack[i]);
                if (previous[i] != merged)
                {
                    previous[i] = merged;
                    changed = true;
                }
            }
            if (changed)
                pending.Enqueue(index);
        }
    }

    /// <summary>Preserves collection references and integer widths at control-flow joins.</summary>
    private static Type MergeTypes(Type? left, Type? right)
    {
        if (IsCollectionReference(left) && IsCollectionReference(right))
        {
            var element = CollectionElement(left);
            if (element is null || element != CollectionElement(right))
                return typeof(MergedCollectionReference);
            if (left!.IsAssignableFrom(right))
            {
                ValidateIndexerConversion(right!, left, element);
                return left;
            }
            if (right!.IsAssignableFrom(left))
            {
                ValidateIndexerConversion(left, right, element);
                return right;
            }
            return typeof(MergedCollectionReference<>).MakeGenericType(element);
        }
        int width = DeviceArithmetic.IntegerWidth(left);
        return width > 0 && width == DeviceArithmetic.IntegerWidth(right)
            ? width == 32 ? typeof(int) : typeof(long) : typeof(object);
    }

    /// <summary>Recognizes source collection references and their merged analysis category.</summary>
    private static bool IsCollectionReference(Type? type) => IsMergedCollectionReference(type) ||
        type is not null && !type.IsValueType && CapturedCollection.ElementType(type) is not null;

    /// <summary>Recognizes analysis markers carrying merged collection references.</summary>
    private static bool IsMergedCollectionReference(Type? type) => type == typeof(MergedCollectionReference) ||
        type is { IsGenericType: true } && type.GetGenericTypeDefinition() == typeof(MergedCollectionReference<>);

    /// <summary>Gets a descriptor's original leaf type, retaining compatible types across control-flow joins.</summary>
    private static Type? CollectionElement(Type? type)
    {
        if (type?.IsByRef == true)
            type = type.GetElementType();
        return type is { IsGenericType: true } && type.GetGenericTypeDefinition() == typeof(MergedCollectionReference<>)
            ? type.GetGenericArguments()[0] : type is null ? null : CapturedCollection.ElementType(type);
    }

    /// <summary>Rejects upcasts whose destination indexer needs a separately captured adapter view.</summary>
    private static void ValidateIndexerConversion(Type source, Type destination, Type element)
    {
        if (GpuCollectionAdapters.ChangesIndexer(source, destination, element))
            throw new NotSupportedException("An implicit collection conversion changes the registered GPU indexer. " +
                "Capture the collection through the required declaration separately.");
    }

    /// <summary>Rejects implicit reference conversions that would reinterpret a mapped collection's element storage.</summary>
    private static void ValidateCollectionConversion(MethodBase method, LoopInstruction instruction, List<Type?> stack)
    {
        void Check(Type? source, Type? destination)
        {
            if (source?.IsByRef == true)
                source = source.GetElementType();
            if (destination?.IsByRef == true)
                destination = destination.GetElementType();
            var element = CollectionElement(destination);
            if (source != typeof(void) && IsCollectionReference(source) && element is not null &&
                CollectionElement(source) != element)
                throw new NotSupportedException("An implicit collection conversion changes the GPU element type. " +
                    "Capture the collection through the required native declaration separately.");
            if (source is not null && IsCollectionReference(source) && destination is not null && element is not null)
                ValidateIndexerConversion(source, destination, element);
        }

        var code = instruction.Code;
        if (instruction.Operand is MethodBase called &&
            (code == OpCodes.Call || code == OpCodes.Callvirt || code == OpCodes.Newobj))
        {
            var parameters = called.GetParameters();
            for (int i = 0; i < parameters.Length; i++)
                Check(stack[stack.Count - parameters.Length + i], parameters[i].ParameterType);
            if (code != OpCodes.Newobj && !called.IsStatic)
            {
                var receiver = stack[stack.Count - parameters.Length - 1];
                // Inherited accessors can use the receiver's registered adapter. A hidden
                // base indexer needs its own capture, even if an optimizer omitted the upcast local.
                if (IsMergedCollectionReference(receiver))
                    Check(receiver, called.DeclaringType);
                else if (receiver is not null && CollectionElement(receiver) is Type element &&
                    called is MethodInfo member && called.DeclaringType is Type declaring &&
                    GpuCollectionAdapters.AccessorOperation(declaring, member, element) is "get_Item" or "set_Item")
                    ValidateIndexerConversion(receiver, declaring, element);
            }
        }
        else if (code.Name!.StartsWith("stloc", StringComparison.Ordinal))
        {
            int local = code == OpCodes.Stloc_0 ? 0 : code == OpCodes.Stloc_1 ? 1 :
                code == OpCodes.Stloc_2 ? 2 : code == OpCodes.Stloc_3 ? 3 : (int)instruction.Operand!;
            Check(stack[^1], method.GetMethodBody()!.LocalVariables[local].LocalType);
        }
        else if (code == OpCodes.Starg || code == OpCodes.Starg_S)
        {
            int parameter = (int)instruction.Operand! - (method.IsStatic ? 0 : 1);
            Check(stack[^1], parameter < 0 ? method.DeclaringType : method.GetParameters()[parameter].ParameterType);
        }
        else if (code == OpCodes.Ret && method is MethodInfo { ReturnType: var returnType } && returnType != typeof(void))
            Check(stack[^1], returnType);
        else if (code == OpCodes.Stind_Ref || code == OpCodes.Stobj)
            Check(stack[^1], code == OpCodes.Stobj ? (Type)instruction.Operand! : stack[^2]?.GetElementType());
        else if (instruction.Operand is FieldInfo field && (code == OpCodes.Stfld || code == OpCodes.Stsfld))
            Check(stack[^1], field.FieldType);
    }

    /// <summary>Rewrites descriptor reference loads/stores, null values, null tests, and collection reference comparisons.</summary>
    /// <returns>True if a replacement was emitted; otherwise false.</returns>
    internal static bool TryEmitReferenceOperation(ILGenerator il, LoopInstruction instruction,
        Type?[] stack, Dictionary<int, Label> labels, Action<ILGenerator> emitContext)
    {
        var code = instruction.Code;
        // References to local collection descriptors point to mapped values, not CLR object slots.
        if (code == OpCodes.Ldind_Ref && stack.Length > 0 && stack[^1]?.IsByRef == true &&
            IsCollectionReference(stack[^1]!.GetElementType()))
        {
            il.Emit(OpCodes.Ldobj, typeof(DeviceCollection));
            return true;
        }
        if (code == OpCodes.Stind_Ref && stack.Length >= 2 && stack[^2]?.IsByRef == true &&
            IsCollectionReference(stack[^2]!.GetElementType()))
        {
            il.Emit(OpCodes.Stobj, typeof(DeviceCollection));
            return true;
        }
        if (code == OpCodes.Ldnull)
        {
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.Null))!);
            emitContext(il);
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.WithContext))!);
            return true;
        }
        if (stack.Length > 0 && IsCollectionReference(stack[^1]) &&
            (code == OpCodes.Brtrue || code == OpCodes.Brtrue_S || code == OpCodes.Brfalse || code == OpCodes.Brfalse_S))
        {
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.IsNull))!);
            il.Emit(code == OpCodes.Brtrue || code == OpCodes.Brtrue_S ? OpCodes.Brfalse : OpCodes.Brtrue,
                labels[(int)instruction.Operand!]);
            return true;
        }
        if (stack.Length >= 2 && (IsCollectionReference(stack[^1]) || IsCollectionReference(stack[^2])) &&
            (code == OpCodes.Ceq || code == OpCodes.Cgt_Un || code == OpCodes.Beq || code == OpCodes.Beq_S ||
             code == OpCodes.Bne_Un || code == OpCodes.Bne_Un_S))
        {
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.Same))!);
            if (code == OpCodes.Cgt_Un)
            {
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Ceq);
            }
            else if (code != OpCodes.Ceq)
                il.Emit(code == OpCodes.Beq || code == OpCodes.Beq_S ? OpCodes.Brtrue : OpCodes.Brfalse,
                    labels[(int)instruction.Operand!]);
            return true;
        }
        return false;
    }

    private static void Apply(MethodBase method, LoopInstruction instruction, List<Type?> stack)
    {
        var code = instruction.Code;
        if (code == OpCodes.Dup)
        {
            stack.Add(stack[^1]);
            return;
        }
        var before = stack.ToArray();
        int pop = Count(code.StackBehaviourPop);
        int push = Count(code.StackBehaviourPush);
        if (instruction.Operand is MethodBase called && code.StackBehaviourPop == StackBehaviour.Varpop)
        {
            pop = called.GetParameters().Length + (code != OpCodes.Newobj && !called.IsStatic ? 1 : 0);
            push = code == OpCodes.Newobj || called is MethodInfo info && info.ReturnType != typeof(void) ? 1 : 0;
        }
        else if (code == OpCodes.Ret)
            pop = method is MethodInfo info && info.ReturnType != typeof(void) ? 1 : 0;
        if (pop > stack.Count)
            throw new NotSupportedException($"Invalid evaluation stack at '{code.Name}'.");
        stack.RemoveRange(stack.Count - pop, pop);
        if (push == 0)
            return;
        Type? type = typeof(object);
        int? argument = LoopIL.Argument(instruction);
        if (argument is not null)
        {
            int parameter = argument.Value - (method.IsStatic ? 0 : 1);
            type = parameter < 0 ? method.DeclaringType : method.GetParameters()[parameter].ParameterType;
        }
        else if (code.Name!.StartsWith("ldloc", StringComparison.Ordinal))
        {
            int local = code == OpCodes.Ldloc_0 ? 0 : code == OpCodes.Ldloc_1 ? 1 :
                code == OpCodes.Ldloc_2 ? 2 : code == OpCodes.Ldloc_3 ? 3 : (int)instruction.Operand!;
            type = method.GetMethodBody()!.LocalVariables[local].LocalType;
            if (code == OpCodes.Ldloca || code == OpCodes.Ldloca_S)
                type = type.MakeByRefType();
        }
        else if (code == OpCodes.Ldarga || code == OpCodes.Ldarga_S)
        {
            int parameter = (int)instruction.Operand! - (method.IsStatic ? 0 : 1);
            type = parameter < 0 ? method.DeclaringType!.MakeByRefType() : method.GetParameters()[parameter].ParameterType.MakeByRefType();
        }
        else if (instruction.Operand is FieldInfo field)
            type = code == OpCodes.Ldflda || code == OpCodes.Ldsflda ? field.FieldType.MakeByRefType() : field.FieldType;
        else if (instruction.Operand is MethodBase call)
            type = code == OpCodes.Newobj ? call.DeclaringType : ((MethodInfo)call).ReturnType;
        else if (code == OpCodes.Ldnull)
            type = typeof(void);
        else if (code == OpCodes.Ldlen)
            type = typeof(int);
        else if (code == OpCodes.Ldelem_Ref && IsMergedCollectionReference(before[^2]))
            type = before[^2];
        else if (code.Name.StartsWith("ldelem.", StringComparison.Ordinal))
            type = before[^2]?.GetElementType() ?? typeof(object);
        else if (code.Name.StartsWith("ldind.", StringComparison.Ordinal))
            type = before[^1]?.GetElementType() ?? typeof(object);
        else if (code == OpCodes.Ldelema)
            type = ((Type)instruction.Operand!).MakeByRefType();
        else if (code == OpCodes.Ldelem || code == OpCodes.Ldobj || code == OpCodes.Castclass || code == OpCodes.Isinst)
            type = (Type)instruction.Operand!;
        else if (code.Name.StartsWith("ldc.i4", StringComparison.Ordinal) ||
            code.Name.StartsWith("conv.i4", StringComparison.Ordinal) || code.Name.StartsWith("conv.u4", StringComparison.Ordinal) ||
            code == OpCodes.Sizeof ||
            code == OpCodes.Conv_I1 || code == OpCodes.Conv_U1 || code == OpCodes.Conv_I2 || code == OpCodes.Conv_U2 ||
            code == OpCodes.Ceq || code == OpCodes.Cgt || code == OpCodes.Cgt_Un || code == OpCodes.Clt || code == OpCodes.Clt_Un)
            type = typeof(int);
        else if (code == OpCodes.Ldc_I8 || code == OpCodes.Conv_I8 || code == OpCodes.Conv_U8)
            type = typeof(long);
        else if (code == OpCodes.Ldc_R4 || code == OpCodes.Conv_R4)
            type = typeof(float);
        else if (code == OpCodes.Ldc_R8 || code == OpCodes.Conv_R8 || code == OpCodes.Conv_R_Un)
            type = typeof(double);
        else if (pop > 0)
            type = before[before.Length - pop];
        for (int i = 0; i < push; i++)
            stack.Add(type);
    }

    private static int Count(StackBehaviour behavior) => behavior is StackBehaviour.Pop0 or StackBehaviour.Push0 or
        StackBehaviour.Varpop or StackBehaviour.Varpush ? 0 : behavior.ToString().Split('_').Length;
}

