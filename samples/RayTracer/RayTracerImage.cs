using StbImageWriteSharp;

namespace RayTracer;

/// <summary>Saves RGB pixels already colored and gamma-corrected on the GPU.</summary>
public static class RayTracerImage
{
    /// <summary>Encodes row-major RGB bytes as a PNG; rendering and color conversion occur on the GPU.</summary>
    public static void SavePng(string path, byte[] pixels, int width, int height)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(pixels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (pixels.Length != checked(width * height * 3))
            throw new ArgumentException("The RGB array must contain exactly width * height * 3 bytes.", nameof(pixels));

        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        using var stream = File.Create(fullPath);
        new ImageWriter().WritePng(pixels, width, height, ColorComponents.RedGreenBlue, stream);
    }
}
