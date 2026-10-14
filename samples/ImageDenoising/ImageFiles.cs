using StbImageSharp;
using StbImageWriteSharp;

namespace ImageDenoising;

/// <summary>Stores a decoded host image before upload or after GPU processing.</summary>
public sealed record RgbImage(int Width, int Height, byte[] Pixels);

/// <summary>Decodes input images and encodes GPU-generated RGB pixels as PNG.</summary>
public static class ImageFiles
{
    /// <summary>Loads an image as three RGB bytes per pixel, in top-to-bottom row order.</summary>
    public static RgbImage Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = File.OpenRead(path);
        ImageResult image;
        try
        {
            image = ImageResult.FromStream(stream, StbImageSharp.ColorComponents.RedGreenBlue);
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            throw new InvalidDataException($"Cannot decode image: {path}.", error);
        }
        ValidatePixels(image.Data, image.Width, image.Height);
        return new(image.Width, image.Height, image.Data);
    }

    /// <summary>Writes an RGB image to PNG, creating its parent directory if needed.</summary>
    public static void SavePng(string path, byte[] pixels, int width, int height)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ValidatePixels(pixels, width, height);
        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        using var stream = File.Create(fullPath);
        new ImageWriter().WritePng(pixels, width, height, StbImageWriteSharp.ColorComponents.RedGreenBlue, stream);
    }

    private static void ValidatePixels(byte[] pixels, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (pixels.Length != checked(width * height * 3))
            throw new ArgumentException("The RGB array must contain exactly width * height * 3 bytes.", nameof(pixels));
    }
}
