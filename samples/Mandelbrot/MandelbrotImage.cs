using StbImageWriteSharp;

namespace Mandelbrot;

/// <summary>Colors escape counts and saves a PNG using StbImageWriteSharp.</summary>
public static class MandelbrotImage
{
    /// <summary>Saves row-major escape counts as a colored RGB image, with interior pixels in black.</summary>
    /// <param name="path">The output file path; parent directories are created when needed.</param>
    /// <param name="iterations">The escape counts, each between zero and maxIterations.</param>
    /// <param name="width">The positive image width.</param>
    /// <param name="height">The positive image height.</param>
    /// <param name="maxIterations">The positive iteration limit used to compute the counts.</param>
    public static void SavePng(string path, int[] iterations, int width, int height, int maxIterations)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(iterations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIterations);
        if (iterations.Length != checked(width * height))
            throw new ArgumentException("The escape-count array must contain exactly width * height elements.", nameof(iterations));

        var pixels = new byte[checked(iterations.Length * 3)];
        double logScale = Math.Log(1.0 + maxIterations);
        for (int i = 0; i < iterations.Length; i++)
        {
            int count = iterations[i];
            if (count < 0 || count > maxIterations)
                throw new ArgumentException("Escape counts must lie between zero and maxIterations.", nameof(iterations));
            var color = ColorPixel(count, maxIterations, logScale);
            pixels[i * 3] = color.R;
            pixels[i * 3 + 1] = color.G;
            pixels[i * 3 + 2] = color.B;
        }

        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        using var stream = File.Create(fullPath);
        new ImageWriter().WritePng(pixels, width, height, ColorComponents.RedGreenBlue, stream);
    }

    /// <summary>Uses a logarithmic blue-green-red palette to make low escape counts visible.</summary>
    private static (byte R, byte G, byte B) ColorPixel(int count, int maxIterations, double logScale)
    {
        if (count == maxIterations)
            return default;

        double t = Math.Log(1.0 + count) / logScale;
        double inverse = 1 - t;
        return (
            (byte)(255 * 9 * inverse * t * t * t),
            (byte)(255 * 15 * inverse * inverse * t * t),
            (byte)(255 * 8.5 * inverse * inverse * inverse * t));
    }
}
