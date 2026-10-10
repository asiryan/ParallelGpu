using System.Runtime.CompilerServices;

namespace ParallelGpu;

/// <summary>Compares host objects by reference identity to preserve aliases.</summary>
internal sealed class ReferenceIdentityComparer : IEqualityComparer<object>
{
    /// <summary>The shared comparer for collection and graph identity registries.</summary>
    internal static readonly ReferenceIdentityComparer Instance = new();

    /// <summary>Creates the singleton identity comparer.</summary>
    private ReferenceIdentityComparer() { }

    /// <summary>Checks whether two values refer to the same object instance.</summary>
    /// <param name="left">The first object reference.</param>
    /// <param name="right">The second object reference.</param>
    /// <returns>True when both references identify the same instance or are both null.</returns>
    public new bool Equals(object? left, object? right) => ReferenceEquals(left, right);

    /// <summary>Returns the identity hash code independently of value equality overrides.</summary>
    /// <param name="value">The object whose identity is hashed.</param>
    /// <returns>The runtime identity hash code.</returns>
    public int GetHashCode(object value) => RuntimeHelpers.GetHashCode(value);
}
