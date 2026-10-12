using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Tracks IL stack types for collection reference comparisons and guarded integer arithmetic.</summary>
internal static class LoopStackTypes
{
    /// <summary>Marks a join of collection references without a shared supported host type.</summary>
    private sealed class MergedCollectionReference;

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
            if (left!.IsAssignableFrom(right))
                return left;
            if (right!.IsAssignableFrom(left))
                return right;
            return typeof(MergedCollectionReference);
        }
        int width = DeviceArithmetic.IntegerWidth(left);
        return width > 0 && width == DeviceArithmetic.IntegerWidth(right)
            ? width == 32 ? typeof(int) : typeof(long) : typeof(object);
    }

    /// <summary>Recognizes source collection references and their merged analysis category.</summary>
    private static bool IsCollectionReference(Type? type) => type == typeof(MergedCollectionReference) ||
        type is not null && !type.IsValueType && CapturedCollection.ElementType(type) is not null;

    /// <summary>Rewrites null values, null tests, and reference comparisons for translated collection descriptors.</summary>
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
        else if (code == OpCodes.Ldelem_Ref && before[^2] == typeof(MergedCollectionReference))
            type = typeof(MergedCollectionReference);
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

