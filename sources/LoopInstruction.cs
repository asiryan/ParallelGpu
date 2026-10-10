using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Stores one decoded instruction from an ordinary managed loop body.</summary>
/// <param name="Offset">The byte offset in the original method body.</param>
/// <param name="Code">The decoded operation code.</param>
/// <param name="Operand">The decoded constant, branch target, resolved member, or null.</param>
internal readonly record struct LoopInstruction(int Offset, OpCode Code, object? Operand);
