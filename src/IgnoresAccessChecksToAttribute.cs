namespace System.Runtime.CompilerServices;

/// <summary>Allows generated assemblies to access private helpers and value fields in a named assembly.</summary>
/// <param name="assemblyName">The simple name of the assembly whose access checks are relaxed.</param>
/// <remarks>The CLR recognizes this attribute by its fully qualified name.</remarks>
[AttributeUsage(AttributeTargets.Assembly, AllowMultiple = true)]
internal sealed class IgnoresAccessChecksToAttribute(string assemblyName) : Attribute
{
    /// <summary>Gets the simple assembly name targeted by the access-check override.</summary>
    public string AssemblyName { get; } = assemblyName;
}
