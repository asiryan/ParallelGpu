namespace RayTracer;

/// <summary>Stores a sphere and its material in the native layout uploaded to the GPU.</summary>
public readonly struct RayTracingSphere
{
    /// <summary>The sphere center in world space.</summary>
    public readonly Float3 Center;
    /// <summary>The positive sphere radius.</summary>
    public readonly float Radius;
    /// <summary>The diffuse reflectance or metal tint in linear RGB.</summary>
    public readonly Float3 Albedo;
    /// <summary>The material kind: zero for diffuse, one for metal, two for glass, three for the checker floor.</summary>
    public readonly int Material;
    /// <summary>The amount of random perturbation applied to a metal reflection.</summary>
    public readonly float Roughness;
    /// <summary>The refractive index of the glass material.</summary>
    public readonly float RefractionIndex;

    /// <summary>Creates a sphere and the parameters used to scatter its incident rays.</summary>
    public RayTracingSphere(Float3 center, float radius, Float3 albedo, int material,
        float roughness = 0, float refractionIndex = 1.5f)
    {
        Center = center;
        Radius = radius;
        Albedo = albedo;
        Material = material;
        Roughness = roughness;
        RefractionIndex = refractionIndex;
    }
}

/// <summary>Stores a pinhole camera and its image-plane vectors for GPU ray generation.</summary>
public readonly struct RayTracingCamera
{
    /// <summary>The camera position in world space.</summary>
    public readonly Float3 Origin;
    /// <summary>The lower-left corner of the image plane.</summary>
    public readonly Float3 LowerLeft;
    /// <summary>The full horizontal extent of the image plane.</summary>
    public readonly Float3 Horizontal;
    /// <summary>The full vertical extent of the image plane.</summary>
    public readonly Float3 Vertical;

    /// <summary>Creates a camera from a position and image-plane geometry.</summary>
    public RayTracingCamera(Float3 origin, Float3 lowerLeft, Float3 horizontal, Float3 vertical)
    {
        Origin = origin;
        LowerLeft = lowerLeft;
        Horizontal = horizontal;
        Vertical = vertical;
    }
}

/// <summary>Builds the host-side scene description that is uploaded once before GPU rendering.</summary>
internal static class RayTracingScene
{
    /// <summary>Creates glass, polished and rough metal, and colorful diffuse spheres on a checker floor.</summary>
    internal static RayTracingSphere[] CreateSpheres() =>
    [
        new(new(0, -1000, 0), 1000, new(0.65f, 0.65f, 0.65f), 3),
        new(new(0, 1, 0), 1, new(1, 1, 1), 2),
        new(new(-2.1f, 0.9f, -0.35f), 0.9f, new(0.95f, 0.72f, 0.28f), 1, 0.08f),
        new(new(2.05f, 0.85f, 0.1f), 0.85f, new(0.9f, 0.93f, 0.98f), 1),
        new(new(-0.8f, 0.65f, -2.4f), 0.65f, new(0.08f, 0.35f, 0.8f), 0),
        new(new(0.85f, 0.6f, -2.1f), 0.6f, new(0.85f, 0.16f, 0.09f), 0),
        new(new(-3.4f, 0.45f, -1.9f), 0.45f, new(0.38f, 0.12f, 0.65f), 0),
        new(new(3.4f, 0.55f, -1.55f), 0.55f, new(0.12f, 0.55f, 0.28f), 0),
        new(new(-1.45f, 0.32f, 1.45f), 0.32f, new(0.72f, 0.78f, 0.82f), 1, 0.4f)
    ];

    /// <summary>Creates a perspective camera with a 38-degree vertical field of view.</summary>
    internal static RayTracingCamera CreateCamera(int width, int height)
    {
        var origin = new Float3(4.5f, 2.8f, 8);
        var target = new Float3(0, 0.75f, -0.5f);
        Float3 backward = Float3.Normalize(origin - target);
        Float3 right = Float3.Normalize(Float3.Cross(new(0, 1, 0), backward));
        Float3 up = Float3.Cross(backward, right);
        float viewportHeight = 2f * MathF.Tan(19f * MathF.PI / 180f);
        Float3 horizontal = right * (viewportHeight * width / height);
        Float3 vertical = up * viewportHeight;
        Float3 lowerLeft = origin - horizontal * 0.5f - vertical * 0.5f - backward;
        return new(origin, lowerLeft, horizontal, vertical);
    }
}
