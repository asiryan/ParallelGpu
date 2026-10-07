using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Decodes and re-emits the managed IL used by supported ordinary loop bodies.</summary>
internal static class LoopIL
{
    /// <summary>Maps encoded instruction values to their reflection operation codes.</summary>
    internal static readonly Dictionary<short, OpCode> Codes = typeof(OpCodes)
        .GetFields(BindingFlags.Public | BindingFlags.Static)
        .Where(field => field.FieldType == typeof(OpCode))
        .Select(field => (OpCode)field.GetValue(null)!)
        .ToDictionary(code => code.Value);

    /// <summary>Decodes a method body and resolves its metadata operands and branch targets.</summary>
    /// <param name="method">The ordinary managed method containing the loop body.</param>
    /// <returns>The decoded instructions in their original order.</returns>
    /// <exception cref="NotSupportedException">The body is unavailable, contains exception handlers, or uses an unsupported operand format.</exception>
    internal static List<LoopInstruction> Read(MethodInfo method)
    {
        var body = method.GetMethodBody() ??
            throw new NotSupportedException("The loop body must have ordinary managed C# code.");
        if (body.ExceptionHandlingClauses.Count != 0)
            throw new NotSupportedException("try/catch/finally is not supported inside GPU loops.");
        var bytes = body.GetILAsByteArray()!;
        var module = method.Module;
        var typeArgs = method.DeclaringType?.GetGenericArguments();
        var methodArgs = method.GetGenericArguments();
        var instructions = new List<LoopInstruction>();
        int position = 0;
        while (position < bytes.Length)
        {
            int offset = position;
            short value = bytes[position++];
            if (value == 0xfe)
                value = (short)(0xfe00 | bytes[position++]);
            var code = Codes[value];
            object? operand;
            switch (code.OperandType)
            {
                case OperandType.InlineNone: operand = null; break;
                case OperandType.ShortInlineI: operand = (sbyte)bytes[position++]; break;
                case OperandType.ShortInlineVar: operand = (int)bytes[position++]; break;
                case OperandType.InlineVar:
                    operand = (int)BitConverter.ToUInt16(bytes, position);
                    position += 2;
                    break;
                case OperandType.InlineI:
                    operand = BitConverter.ToInt32(bytes, position);
                    position += 4;
                    break;
                case OperandType.InlineI8:
                    operand = BitConverter.ToInt64(bytes, position);
                    position += 8;
                    break;
                case OperandType.ShortInlineR:
                    operand = BitConverter.ToSingle(bytes, position);
                    position += 4;
                    break;
                case OperandType.InlineR:
                    operand = BitConverter.ToDouble(bytes, position);
                    position += 8;
                    break;
                // Branch displacements are relative to the end of their encoded instruction.
                case OperandType.ShortInlineBrTarget:
                    operand = position + 1 + (sbyte)bytes[position];
                    position++;
                    break;
                case OperandType.InlineBrTarget:
                    operand = position + 4 + BitConverter.ToInt32(bytes, position);
                    position += 4;
                    break;
                case OperandType.InlineSwitch:
                    int count = BitConverter.ToInt32(bytes, position);
                    position += 4;
                    int end = position + count * 4;
                    var targets = new int[count];
                    for (int i = 0; i < count; i++, position += 4)
                        targets[i] = end + BitConverter.ToInt32(bytes, position);
                    operand = targets;
                    break;
                case OperandType.InlineField:
                case OperandType.InlineMethod:
                case OperandType.InlineType:
                case OperandType.InlineTok:
                    int token = BitConverter.ToInt32(bytes, position);
                    position += 4;
                    operand = module.ResolveMember(token, typeArgs, methodArgs);
                    break;
                default:
                    throw new NotSupportedException($"GPU loop instruction '{code.Name}' is not supported.");
            }
            instructions.Add(new(offset, code, operand));
        }
        return instructions;
    }

    /// <summary>Recognizes an argument-load instruction and extracts its argument index.</summary>
    /// <param name="instruction">The decoded instruction to inspect.</param>
    /// <returns>The loaded argument index, or null for a different instruction.</returns>
    internal static int? Argument(LoopInstruction instruction)
    {
        var code = instruction.Code;
        if (code == OpCodes.Ldarg_0) return 0;
        if (code == OpCodes.Ldarg_1) return 1;
        if (code == OpCodes.Ldarg_2) return 2;
        if (code == OpCodes.Ldarg_3) return 3;
        if (code == OpCodes.Ldarg || code == OpCodes.Ldarg_S)
            return (int)instruction.Operand!;
        return null;
    }

    /// <summary>Re-emits a decoded instruction using translated labels and resolved operands.</summary>
    /// <param name="il">The destination IL generator.</param>
    /// <param name="instruction">The decoded source instruction.</param>
    /// <param name="labels">Destination labels indexed by original instruction offsets.</param>
    internal static void Emit(ILGenerator il, LoopInstruction instruction, Dictionary<int, Label> labels)
    {
        var code = instruction.Code;
        // Replacements may exceed a short branch's range, so emit its long form and let labels resolve it.
        if (code.OperandType == OperandType.ShortInlineBrTarget)
            code = Codes.Values.Single(candidate => candidate.Name == code.Name![..^2]);
        switch (instruction.Code.OperandType)
        {
            case OperandType.InlineNone: il.Emit(code); break;
            case OperandType.ShortInlineBrTarget:
            case OperandType.InlineBrTarget: il.Emit(code, labels[(int)instruction.Operand!]); break;
            case OperandType.InlineSwitch:
                il.Emit(code, ((int[])instruction.Operand!).Select(target => labels[target]).ToArray());
                break;
            default:
                switch (instruction.Operand)
                {
                    case sbyte value: il.Emit(code, value); break;
                    case int value when code.OperandType is OperandType.InlineVar or OperandType.ShortInlineVar:
                        il.Emit(code, (short)value); break;
                    case int value: il.Emit(code, value); break;
                    case long value: il.Emit(code, value); break;
                    case float value: il.Emit(code, value); break;
                    case double value: il.Emit(code, value); break;
                    case FieldInfo field: il.Emit(code, field); break;
                    case MethodInfo method: il.Emit(code, method); break;
                    case ConstructorInfo constructor: il.Emit(code, constructor); break;
                    case Type type: il.Emit(code, type); break;
                    default: throw new NotSupportedException($"GPU loop instruction '{code.Name}' is not supported.");
                }
                break;
        }
    }
}
