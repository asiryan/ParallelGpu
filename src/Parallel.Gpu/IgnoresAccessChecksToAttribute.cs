namespace System.Runtime.CompilerServices;

// The CLR recognizes this attribute on generated assemblies. It lets a rewritten
// body call the same private scalar helpers that its original C# body could call.
[AttributeUsage(AttributeTargets.Assembly, AllowMultiple = true)]
internal sealed class IgnoresAccessChecksToAttribute(string assemblyName) : Attribute
{
    public string AssemblyName { get; } = assemblyName;
}
