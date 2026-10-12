using ParallelGpu;

namespace Mandelbrot;

/// <summary>Computes Mandelbrot escape counts with one independent iteration per pixel.</summary>
public static class MandelbrotRenderer
{
    /// <summary>Computes a row-major escape-count image using an automatically compiled GPU lambda.</summary>
    /// <param name="gpu">The executor used to compile the loop and transfer its results.</param>
    /// <param name="width">The positive image width.</param>
    /// <param name="height">The positive image height.</param>
    /// <param name="maxIterations">The positive limit; pixels reaching it are treated as interior points.</param>
    /// <returns>One escape count per pixel, ordered from the top-left corner by row.</returns>
    /// <remarks>The horizontal range is [-2.5, 1]; the vertical range preserves square pixels around Im(c) = 0.</remarks>
    public static int[] RenderOnGpu(Gpu gpu, int width, int height, int maxIterations)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        int length = ValidateDimensions(width, height, maxIterations);
        var iterations = new int[length];
        double pixelScale = 3.5 / width;

        gpu.For<int>(0, iterations.Length, index =>
            iterations[index] = CountIterations(index, width, height, pixelScale, maxIterations));

        return iterations;
    }

    /// <summary>Computes the same escape counts using CPU Parallel.For.</summary>
    /// <param name="width">The positive image width.</param>
    /// <param name="height">The positive image height.</param>
    /// <param name="maxIterations">The positive iteration limit.</param>
    /// <returns>One escape count per pixel in row-major order.</returns>
    public static int[] RenderOnCpu(int width, int height, int maxIterations)
    {
        int length = ValidateDimensions(width, height, maxIterations);
        var iterations = new int[length];
        double pixelScale = 3.5 / width;

        Parallel.For(0, iterations.Length, index =>
            iterations[index] = CountIterations(index, width, height, pixelScale, maxIterations));

        return iterations;
    }

    /// <summary>Validates host arguments before allocating memory or launching a loop.</summary>
    private static int ValidateDimensions(int width, int height, int maxIterations)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIterations);
        return checked(width * height);
    }

    /// <summary>Samples a pixel center and iterates z = z² + c until escape or the configured limit.</summary>
    private static int CountIterations(int index, int width, int height, double pixelScale, int maxIterations)
    {
        int x = index % width;
        int y = index / width;
        double real = -2.5 + (x + 0.5) * pixelScale;
        double imaginary = (height / 2.0 - y - 0.5) * pixelScale;
        double zReal = 0;
        double zImaginary = 0;
        int count = 0;

        while (zReal * zReal + zImaginary * zImaginary <= 4 && count < maxIterations)
        {
            double nextReal = zReal * zReal - zImaginary * zImaginary + real;
            zImaginary = 2 * zReal * zImaginary + imaginary;
            zReal = nextReal;
            count++;
        }

        return count;
    }
}
