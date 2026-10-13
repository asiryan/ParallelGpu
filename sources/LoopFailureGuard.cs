using System.Reflection;
using System.Reflection.Emit;
using ILGPU;

namespace ParallelGpu;

/// <summary>Stops translated control flow after an operation records a deferred device failure.</summary>
internal sealed class LoopFailureGuard
{
    /// <summary>The generated volatile reader of the invocation's shared error flag.</summary>
    private readonly MethodInfo hasFailed;

    /// <summary>Defines an error reader in the same generated type as translated helpers.</summary>
    /// <param name="helpers">The type containing translated scalar and collection operations.</param>
    internal LoopFailureGuard(TypeBuilder helpers)
    {
        var method = helpers.DefineMethod("HasFailed", MethodAttributes.Public | MethodAttributes.Static,
            typeof(bool), [typeof(DeviceExecutionContext)]);
        hasFailed = method;
        var il = method.GetILGenerator();
        il.Emit(OpCodes.Ldarga_S, (byte)0);
        il.Emit(OpCodes.Ldflda, typeof(DeviceExecutionContext).GetField(nameof(DeviceExecutionContext.Errors))!);
        il.Emit(OpCodes.Ldc_I8, 0L);
        il.Emit(OpCodes.Call, typeof(ArrayView<int>).GetMethod("get_Item", [typeof(long)])!);
        // The shared flag is written atomically. A volatile read avoids a contended atomic
        // operation on every loop backedge while preventing a cached zero from extending a loop.
        il.Emit(OpCodes.Volatile);
        il.Emit(OpCodes.Ldind_I4);
        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Cgt_Un);
        il.Emit(OpCodes.Ret);
    }

    /// <summary>Emits a stack-preserving continuation or a safe early return after a deferred failure.</summary>
    /// <param name="il">The destination generator.</param>
    /// <param name="stackCount">The number of values currently on the evaluation stack.</param>
    /// <param name="returnType">The mapped result type of the enclosing body or helper.</param>
    /// <param name="emitContext">Emits the enclosing invocation's context.</param>
    /// <param name="parameters">The mapped helper arguments, used for safe reference returns.</param>
    internal void Emit(ILGenerator il, int stackCount, Type returnType,
        Action<ILGenerator> emitContext, IReadOnlyList<Type>? parameters = null)
    {
        var success = il.DefineLabel();
        emitContext(il);
        il.Emit(OpCodes.Call, hasFailed);
        il.Emit(OpCodes.Brfalse, success);
        for (int i = 0; i < stackCount; i++)
            il.Emit(OpCodes.Pop);
        if (returnType.IsByRef)
        {
            int argument = -1;
            if (parameters is not null)
                for (int i = 0; i < parameters.Count; i++)
                    if (parameters[i] == returnType)
                    {
                        argument = i;
                        break;
                    }
            if (argument >= 0)
                il.Emit(OpCodes.Ldarg, (short)argument);
            else
            {
                var context = il.DeclareLocal(typeof(DeviceExecutionContext));
                emitContext(il);
                il.Emit(OpCodes.Stloc, context);
                il.Emit(OpCodes.Ldloca, context);
                il.Emit(OpCodes.Call, typeof(DeviceExecutionContext)
                    .GetMethod(nameof(DeviceExecutionContext.ScratchAddress))!
                    .MakeGenericMethod(returnType.GetElementType()!));
            }
        }
        else if (returnType != typeof(void))
        {
            var result = il.DeclareLocal(returnType);
            il.Emit(OpCodes.Ldloca, result);
            il.Emit(OpCodes.Initobj, returnType);
            il.Emit(OpCodes.Ldloc, result);
        }
        il.Emit(OpCodes.Ret);
        il.MarkLabel(success);
    }

    /// <summary>Recognizes branches whose target can repeat previously executed instructions.</summary>
    internal static bool IsBackwardBranch(LoopInstruction instruction) =>
        instruction.Code == OpCodes.Switch
            ? ((int[])instruction.Operand!).Any(offset => offset <= instruction.Offset)
            : instruction.Code.OperandType is OperandType.InlineBrTarget or OperandType.ShortInlineBrTarget &&
                (int)instruction.Operand! <= instruction.Offset;

    /// <summary>Retains post-operation exits for integer division and remainder, which can record deferred errors.</summary>
    /// <param name="instruction">The source arithmetic instruction already lowered by the translator.</param>
    /// <param name="stack">The source evaluation-stack types before the instruction.</param>
    /// <returns>True for guarded integer division or remainder; false for pure arithmetic replacements.</returns>
    internal static bool RequiredAfterArithmetic(LoopInstruction instruction, Type?[] stack) =>
        DeviceArithmetic.IsDivision(instruction.Code) && stack.Length >= 2 &&
        DeviceArithmetic.IntegerWidth(stack[^2]) > 0;

    /// <summary>Omits error reads after known pure calls while retaining exits after unknown helpers and constructors.</summary>
    /// <param name="method">The original method or constructor emitted by the translator.</param>
    /// <returns>False only for native buffer accessors that cannot record automatic-loop errors.</returns>
    internal static bool RequiredAfterCall(MethodBase method)
    {
        if (method is MethodInfo info && info.DeclaringType is { IsGenericType: true } declaring &&
            declaring.GetGenericTypeDefinition() == typeof(GpuView<>) && !info.IsStatic &&
            (DeviceNativeTypes.IsElementReference(info) ||
             info.Name == "get_Length" && info.ReturnType == typeof(int) && info.GetParameters().Length == 0))
            return false;
        return true;
    }

    /// <summary>Computes the evaluation-stack size after an operation that may record a failure.</summary>
    internal static int StackAfter(LoopInstruction instruction, int before)
    {
        if (instruction.Operand is MethodBase method &&
            (instruction.Code == OpCodes.Call || instruction.Code == OpCodes.Callvirt || instruction.Code == OpCodes.Newobj))
            return before - method.GetParameters().Length -
                (instruction.Code != OpCodes.Newobj && !method.IsStatic ? 1 : 0) +
                (instruction.Code == OpCodes.Newobj || method is MethodInfo info && info.ReturnType != typeof(void) ? 1 : 0);
        return before - PopCount(instruction.Code.StackBehaviourPop) + PushCount(instruction.Code.StackBehaviourPush);
    }

    /// <summary>Counts operands consumed by fixed-stack operations.</summary>
    private static int PopCount(StackBehaviour behaviour) => behaviour switch
    {
        StackBehaviour.Pop0 => 0,
        StackBehaviour.Pop1 or StackBehaviour.Popi or StackBehaviour.Popref => 1,
        StackBehaviour.Pop1_pop1 or StackBehaviour.Popi_pop1 or StackBehaviour.Popi_popi or
            StackBehaviour.Popi_popi8 or StackBehaviour.Popi_popr4 or StackBehaviour.Popi_popr8 or
            StackBehaviour.Popref_pop1 or StackBehaviour.Popref_popi => 2,
        StackBehaviour.Popi_popi_popi or StackBehaviour.Popref_popi_pop1 or StackBehaviour.Popref_popi_popi or
            StackBehaviour.Popref_popi_popi8 or StackBehaviour.Popref_popi_popr4 or StackBehaviour.Popref_popi_popr8 or
            StackBehaviour.Popref_popi_popref => 3,
        _ => throw new NotSupportedException($"Unsupported guarded stack behavior: {behaviour}.")
    };

    /// <summary>Counts results produced by fixed-stack operations.</summary>
    private static int PushCount(StackBehaviour behaviour) => behaviour switch
    {
        StackBehaviour.Push0 => 0,
        StackBehaviour.Push1 or StackBehaviour.Pushi or StackBehaviour.Pushi8 or
            StackBehaviour.Pushr4 or StackBehaviour.Pushr8 or StackBehaviour.Pushref => 1,
        StackBehaviour.Push1_push1 => 2,
        _ => throw new NotSupportedException($"Unsupported guarded stack behavior: {behaviour}.")
    };
}
