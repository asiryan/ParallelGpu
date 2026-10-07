using System.Runtime.CompilerServices;

namespace ParallelGpu;

internal sealed class ReferenceIdentityComparer : IEqualityComparer<object>
{
    internal static readonly ReferenceIdentityComparer Instance = new();

    private ReferenceIdentityComparer() { }

    public new bool Equals(object? left, object? right) => ReferenceEquals(left, right);

    public int GetHashCode(object value) => RuntimeHelpers.GetHashCode(value);
}
