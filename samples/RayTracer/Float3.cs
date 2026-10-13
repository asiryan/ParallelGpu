namespace RayTracer;

/// <summary>Stores a three-component vector or linear RGB color in a device-compatible value type.</summary>
public readonly struct Float3
{
    /// <summary>The first component.</summary>
    public readonly float X;
    /// <summary>The second component.</summary>
    public readonly float Y;
    /// <summary>The third component.</summary>
    public readonly float Z;

    /// <summary>Creates a vector from its three components.</summary>
    public Float3(float x, float y, float z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    /// <summary>Adds two vectors component by component.</summary>
    public static Float3 operator +(Float3 left, Float3 right) =>
        new(left.X + right.X, left.Y + right.Y, left.Z + right.Z);

    /// <summary>Subtracts two vectors component by component.</summary>
    public static Float3 operator -(Float3 left, Float3 right) =>
        new(left.X - right.X, left.Y - right.Y, left.Z - right.Z);

    /// <summary>Reverses a vector.</summary>
    public static Float3 operator -(Float3 value) => new(-value.X, -value.Y, -value.Z);

    /// <summary>Scales all vector components.</summary>
    public static Float3 operator *(Float3 value, float scale) =>
        new(value.X * scale, value.Y * scale, value.Z * scale);

    /// <summary>Multiplies two colors component by component.</summary>
    public static Float3 operator *(Float3 left, Float3 right) =>
        new(left.X * right.X, left.Y * right.Y, left.Z * right.Z);

    /// <summary>Computes the dot product.</summary>
    public static float Dot(Float3 left, Float3 right) =>
        left.X * right.X + left.Y * right.Y + left.Z * right.Z;

    /// <summary>Computes the cross product.</summary>
    public static Float3 Cross(Float3 left, Float3 right) => new(
        left.Y * right.Z - left.Z * right.Y,
        left.Z * right.X - left.X * right.Z,
        left.X * right.Y - left.Y * right.X);

    /// <summary>Returns a unit vector, or zero if the original vector is too small to normalize.</summary>
    public static Float3 Normalize(Float3 value)
    {
        float squaredLength = Dot(value, value);
        return squaredLength > 1e-20f ? value * (1f / MathF.Sqrt(squaredLength)) : default;
    }
}
