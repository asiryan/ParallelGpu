using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Validates read-only use of captured-value addresses in loop IL.</summary>
internal static class ReadOnlyCapture
{
    /// <summary>Checks the consumption of an address loaded from a closure value.</summary>
    /// <param name="instructions">The original loop instructions.</param>
    /// <param name="position">The instruction that loads the captured value's address.</param>
    /// <param name="collection">Whether indexed writes through the captured view are allowed.</param>
    /// <exception cref="NotSupportedException">The captured value itself can be modified or its address can escape.</exception>
    internal static void Validate(List<LoopInstruction> instructions, int position, bool collection = false)
    {
        // Follow branches while the address is live, tracking only the stack suffix above this capture.
        var offsets = instructions.Select((instruction, index) => (instruction.Offset, index))
            .ToDictionary(item => item.Offset, item => item.index);
        var pending = new Queue<(int Position, List<bool> Stack, HashSet<int> Locals)>();
        var visited = new HashSet<string>();
        pending.Enqueue((position + 1, new List<bool> { true }, new HashSet<int>()));
        while (pending.Count > 0)
        {
            var (current, stack, locals) = pending.Dequeue();
            if (!stack.Contains(true) && locals.Count == 0)
                continue;
            if (current >= instructions.Count)
                throw Unsupported();
            string key = current + ":" + string.Concat(stack.Select(value => value ? '1' : '0')) +
                ":" + string.Join(",", locals.OrderBy(value => value));
            if (!visited.Add(key))
                continue;
            var instruction = instructions[current];
            var code = instruction.Code;
            if (collection && code.FlowControl == FlowControl.Return && !stack.Contains(true))
                continue;
            if (code.FlowControl is FlowControl.Return or FlowControl.Throw)
                throw Unsupported();
            if (collection && code.Name!.StartsWith("stloc", StringComparison.Ordinal))
            {
                int local = Local(instruction);
                if (Pop(stack)) locals.Add(local); else locals.Remove(local);
            }
            else if (collection && code.Name!.StartsWith("ldloc", StringComparison.Ordinal))
                stack.Add(locals.Contains(Local(instruction)));
            else if (code == OpCodes.Dup)
                stack.Add(stack[^1]);
            else if (instruction.Operand is MethodBase called &&
                (code == OpCodes.Call || code == OpCodes.Callvirt || code == OpCodes.Newobj))
            {
                foreach (var parameter in called.GetParameters())
                    if (Pop(stack))
                        throw Unsupported(); // A ref helper can mutate the capture even with a void result.
                if (code != OpCodes.Newobj && !called.IsStatic && Pop(stack))
                {
                    if (called is not MethodInfo method ||
                        !(collection && IsCollectionMember(method)) &&
                        ((method.ReturnType.IsByRef && !DeviceNativeTypes.IsElementReference(method)) ||
                            !IsReadOnly(method, new HashSet<MethodInfo>())))
                        throw Unsupported();
                }
                if (code == OpCodes.Newobj || called is MethodInfo info && info.ReturnType != typeof(void))
                    stack.Add(false);
            }
            else if (code == OpCodes.Ldflda)
            {
                bool capture = Pop(stack);
                stack.Add(capture);
            }
            else
            {
                bool usesCapture = false;
                for (int j = 0; j < Count(code.StackBehaviourPop); j++)
                    usesCapture |= Pop(stack);
                // Reading through an address produces a value copy. Other uses can write or escape it.
                if (usesCapture && code != OpCodes.Ldfld && code != OpCodes.Ldobj &&
                    !code.Name!.StartsWith("ldind", StringComparison.Ordinal))
                    throw Unsupported();
                for (int j = 0; j < Count(code.StackBehaviourPush); j++)
                    stack.Add(false);
            }
            if (code.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch)
            {
                var targets = instruction.Operand is int[] choices ? choices : [(int)instruction.Operand!];
                foreach (int target in targets)
                    pending.Enqueue((offsets[target], new List<bool>(stack), new HashSet<int>(locals)));
                if (code.FlowControl == FlowControl.Branch)
                    continue;
            }
            pending.Enqueue((current + 1, stack, locals));
        }
    }

    private static int Local(LoopInstruction instruction) => instruction.Code.Name! switch
    {
        "ldloc.0" or "stloc.0" => 0,
        "ldloc.1" or "stloc.1" => 1,
        "ldloc.2" or "stloc.2" => 2,
        "ldloc.3" or "stloc.3" => 3,
        _ => (int)instruction.Operand!
    };

    private static bool IsCollectionMember(MethodInfo method)
    {
        var element = CapturedCollection.ElementType(method.DeclaringType!);
        return element is not null &&
            (GpuCollectionAdapters.AccessorOperation(method.DeclaringType!, method, element) is not null ||
                method.Name is "get_Item" or "set_Item" or "get_Count" or "get_Length" or
                    "get_Span" or "get_IsEmpty" or "Slice");
    }
    /// <summary>Pops a tracked reference, treating stack values below the capture as unrelated.</summary>
    /// <param name="stack">The tracked suffix of the evaluation stack.</param>
    /// <returns>True if the popped value is derived from the capture address.</returns>
    private static bool Pop(List<bool> stack)
    {
        if (stack.Count == 0)
            return false;
        bool result = stack[^1];
        stack.RemoveAt(stack.Count - 1);
        return result;
    }

    /// <summary>Gets the operand count for a fixed stack behavior.</summary>
    /// <param name="behavior">The opcode's stack behavior.</param>
    /// <returns>The number of consumed or produced stack operands.</returns>
    private static int Count(StackBehaviour behavior) => behavior is StackBehaviour.Pop0 or StackBehaviour.Push0
        ? 0 : behavior.ToString().Split('_').Length;

    /// <summary>Recognizes readonly members and helpers that do not write or expose receiver addresses.</summary>
    /// <param name="method">The instance method consuming the capture address.</param>
    /// <param name="visited">The inspected method call graph.</param>
    /// <returns>True for members marked readonly or passing the receiver-mutation checks.</returns>
    private static bool IsReadOnly(MethodInfo method, HashSet<MethodInfo> visited)
    {
        const string attribute = "System.Runtime.CompilerServices.IsReadOnlyAttribute";
        if (method.CustomAttributes.Any(item => item.AttributeType.FullName == attribute) ||
            method.DeclaringType!.CustomAttributes.Any(item => item.AttributeType.FullName == attribute))
            return true;
        var body = method.GetMethodBody();
        if (!visited.Add(method) || body is null || body.ExceptionHandlingClauses.Count != 0)
            return false;
        try
        {
            return HasReadOnlyReceiver(method, LoopIL.Read(method), visited);
        }
        catch (NotSupportedException)
        {
            return false;
        }
    }

    /// <summary>Tracks receiver-derived addresses separately from independent value copies at each IL offset.</summary>
    private static bool HasReadOnlyReceiver(MethodInfo method, List<LoopInstruction> instructions,
        HashSet<MethodInfo> visited)
    {
        if (instructions.Count == 0 || method.ReturnType.IsByRef)
            return false;
        var offsets = instructions.Select((instruction, index) => (instruction.Offset, index))
            .ToDictionary(item => item.Offset, item => item.index);
        var states = new Dictionary<int, (bool[] Stack, bool[] Locals, bool[] Arguments)>();
        var arguments = new bool[method.GetParameters().Length + 1];
        arguments[0] = true;
        states.Add(0, (Array.Empty<bool>(), new bool[method.GetMethodBody()!.LocalVariables.Count], arguments));
        var pending = new Queue<int>();
        pending.Enqueue(0);
        while (pending.Count > 0)
        {
            int index = pending.Dequeue();
            var state = states[index];
            var stack = state.Stack.ToList();
            var locals = (bool[])state.Locals.Clone();
            var currentArguments = (bool[])state.Arguments.Clone();
            var instruction = instructions[index];
            var code = instruction.Code;
            bool Take()
            {
                if (stack.Count == 0)
                    throw Unsupported();
                return Pop(stack);
            }

            int? argument = LoopIL.Argument(instruction);
            if (argument is not null)
                stack.Add(currentArguments[argument.Value]);
            else if (code == OpCodes.Ldarga || code == OpCodes.Ldarga_S)
                stack.Add(currentArguments[(int)instruction.Operand!]);
            else if (code == OpCodes.Starg || code == OpCodes.Starg_S)
                currentArguments[(int)instruction.Operand!] = Take();
            else if (code.Name!.StartsWith("stloc", StringComparison.Ordinal))
                locals[Local(instruction)] = Take();
            else if (code == OpCodes.Ldloca || code == OpCodes.Ldloca_S)
            {
                // The address of a byref local could indirectly expose its receiver alias.
                if (locals[Local(instruction)])
                    return false;
                stack.Add(false);
            }
            else if (code.Name.StartsWith("ldloc", StringComparison.Ordinal))
                stack.Add(locals[Local(instruction)]);
            else if (code == OpCodes.Dup)
            {
                bool value = Take();
                stack.Add(value);
                stack.Add(value);
            }
            else if (code == OpCodes.Pop)
                Take();
            else if (code == OpCodes.Ldflda)
                stack.Add(Take());
            else if (code == OpCodes.Ldfld || code == OpCodes.Ldobj ||
                code.Name.StartsWith("ldind", StringComparison.Ordinal))
            {
                Take();
                stack.Add(false); // Loading a supported value copies it; it does not retain its source address.
            }
            else if (code == OpCodes.Stfld || code == OpCodes.Stobj ||
                code.Name.StartsWith("stind", StringComparison.Ordinal))
            {
                bool value = Take();
                bool destination = Take();
                if (value || destination)
                    return false;
            }
            else if (code == OpCodes.Cpobj)
            {
                Take(); // Copying the receiver payload into independent storage is a read.
                if (Take())
                    return false;
            }
            else if (code == OpCodes.Initobj)
            {
                if (Take())
                    return false;
            }
            else if (instruction.Operand is MethodBase called &&
                (code == OpCodes.Call || code == OpCodes.Callvirt || code == OpCodes.Newobj))
            {
                foreach (var parameter in called.GetParameters())
                    if (Take())
                        return false; // Receiver addresses cannot escape through helper parameters.
                if (code != OpCodes.Newobj && !called.IsStatic && Take())
                {
                    if (called is not MethodInfo instance ||
                        instance.ReturnType.IsByRef && !DeviceNativeTypes.IsElementReference(instance) ||
                        !IsReadOnly(instance, new HashSet<MethodInfo>(visited)))
                        return false;
                }
                if (code == OpCodes.Newobj || called is MethodInfo info && info.ReturnType != typeof(void))
                    stack.Add(false);
            }
            else if (code == OpCodes.Ret)
            {
                if (method.ReturnType != typeof(void) && Take())
                    return false;
            }
            else
            {
                if (code.StackBehaviourPop == StackBehaviour.Varpop || code.StackBehaviourPush == StackBehaviour.Varpush)
                    return false;
                for (int i = 0; i < Count(code.StackBehaviourPop); i++)
                    if (Take())
                        return false;
                for (int i = 0; i < Count(code.StackBehaviourPush); i++)
                    stack.Add(false);
            }

            if (code.FlowControl is FlowControl.Return or FlowControl.Throw)
                continue;
            if (code.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch)
            {
                foreach (int target in instruction.Operand is int[] choices ? choices : [(int)instruction.Operand!])
                    if (!offsets.TryGetValue(target, out int targetIndex) || !Merge(targetIndex, stack, locals, currentArguments))
                        return false;
                if (code.FlowControl == FlowControl.Branch)
                    continue;
            }
            if (index + 1 >= instructions.Count || !Merge(index + 1, stack, locals, currentArguments))
                return false;
        }
        return true;

        bool Merge(int index, List<bool> stack, bool[] locals, bool[] currentArguments)
        {
            if (!states.TryGetValue(index, out var previous))
            {
                states.Add(index, (stack.ToArray(), (bool[])locals.Clone(), (bool[])currentArguments.Clone()));
                pending.Enqueue(index);
                return true;
            }
            if (previous.Stack.Length != stack.Count)
                return false;
            bool changed = false;
            MergeValues(previous.Stack, stack);
            MergeValues(previous.Locals, locals);
            MergeValues(previous.Arguments, currentArguments);
            if (changed)
                pending.Enqueue(index);
            return true;

            void MergeValues(bool[] destination, IReadOnlyList<bool> source)
            {
                for (int i = 0; i < destination.Length; i++)
                    if (source[i] && !destination[i])
                    {
                        destination[i] = true;
                        changed = true;
                    }
            }
        }
    }

    /// <summary>Creates the diagnostic for a writable or escaping captured-value address.</summary>
    /// <returns>The unsupported-capture diagnostic.</returns>
    private static NotSupportedException Unsupported() => new(
        "Captured values are read-only inside GPU loops. Copy the value to a local variable or update a collection element.");
}
