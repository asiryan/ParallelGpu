using System.Numerics;
using ParallelGpu;

namespace FastFourierTransform;

/// <summary>Computes a forward two-dimensional radix-2 Cooley–Tukey FFT using complex GPU buffers.</summary>
public static class FourierTransform
{
    /// <summary>Transforms every row and then every column, keeping intermediate values on the device.</summary>
    /// <param name="gpu">The executor used for all FFT passes.</param>
    /// <param name="input">The complex matrix, which remains unchanged. Both dimensions must be positive powers of two.</param>
    /// <param name="normalized">Whether to multiply the spectrum by 1 / sqrt(rows * columns), as in UMapx.</param>
    /// <returns>A new matrix with the input shape, in natural frequency order, using a negative exponential on both axes.</returns>
    /// <exception cref="ArgumentNullException">The executor or input is null.</exception>
    /// <exception cref="ArgumentException">Either input dimension is empty or is not a power of two.</exception>
    public static Complex[,] Forward2D(Gpu gpu, Complex[,] input, bool normalized = true)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        ArgumentNullException.ThrowIfNull(input);
        int rows = input.GetLength(0);
        int columns = input.GetLength(1);
        if (!IsPowerOfTwo(rows) || !IsPowerOfTwo(columns))
            throw new ArgumentException("Both matrix dimensions must be positive powers of two.", nameof(input));

        // Flatten only for transfer; FFT stages run on the GPU.
        var hostValues = new Complex[input.Length];
        int index = 0;
        foreach (Complex value in input)
            hostValues[index++] = value;

        using var buffer = gpu.Upload(hostValues);
        var values = buffer.DeviceView;
        Transform(gpu, values, columns, rows, elementStride: 1, batchStride: columns);
        Transform(gpu, values, rows, columns, elementStride: columns, batchStride: 1);

        if (normalized)
        {
            var scale = new Complex(1 / Math.Sqrt(hostValues.Length), 0);
            gpu.For(0, values.Length, i => values[i] *= scale);
        }

        buffer.CopyTo(hostValues);
        var result = new Complex[rows, columns];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                result[row, column] = hostValues[row * columns + column];
        return result;
    }

    /// <summary>Runs independent strided one-dimensional FFTs within the same device buffer.</summary>
    /// <param name="gpu">The executor that owns the device view.</param>
    /// <param name="values">The row-major matrix values.</param>
    /// <param name="length">The power-of-two length of each transform.</param>
    /// <param name="batchCount">The number of independent transforms.</param>
    /// <param name="elementStride">The distance between values in one transform.</param>
    /// <param name="batchStride">The distance between the starts of adjacent transforms.</param>
    private static void Transform(Gpu gpu, GpuView<Complex> values, int length, int batchCount,
        int elementStride, int batchStride)
    {
        if (length == 1)
            return;

        int levels = 0;
        for (int remaining = length; remaining > 1; remaining >>= 1)
            levels++;

        // Swap each bit-reversed pair exactly once; iterations own disjoint pairs.
        gpu.For(0, batchCount * length, index =>
        {
            int batch = index / length;
            int position = index % length;
            int reversed = ReverseBits(position, levels);
            if (position < reversed)
            {
                int left = batch * batchStride + position * elementStride;
                int right = batch * batchStride + reversed * elementStride;
                Complex temporary = values[left];
                values[left] = values[right];
                values[right] = temporary;
            }
        });

        using var twiddleBuffer = gpu.Allocate<Complex>(length / 2);
        var twiddles = twiddleBuffer.DeviceView;
        double angleStep = -2 * Math.PI / length;
        gpu.For(0, twiddles.Length, index =>
        {
            double angle = angleStep * index;
            twiddles[index] = new Complex(Math.Cos(angle), Math.Sin(angle));
        });

        int butterfliesPerBatch = length / 2;
        for (int size = 2; ; size <<= 1)
        {
            int half = size / 2;
            int twiddleStep = length / size;
            gpu.For(0, batchCount * butterfliesPerBatch, index =>
            {
                int batch = index / butterfliesPerBatch;
                int butterfly = index % butterfliesPerBatch;
                int offset = butterfly % half;
                int position = butterfly / half * size + offset;
                int left = batch * batchStride + position * elementStride;
                int right = left + half * elementStride;
                Complex even = values[left];
                Complex odd = values[right] * twiddles[offset * twiddleStep];
                values[left] = even + odd;
                values[right] = even - odd;
            });

            // Each synchronous For completes a stage before the next one reads it.
            if (size == length)
                break;
        }
    }

    /// <summary>Reverses exactly the low-order bits used to index one transform.</summary>
    /// <param name="value">The element index.</param>
    /// <param name="levels">The number of index bits.</param>
    /// <returns>The index in bit-reversed order.</returns>
    private static int ReverseBits(int value, int levels)
    {
        int reversed = 0;
        for (int bit = 0; bit < levels; bit++)
        {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
        }
        return reversed;
    }

    /// <summary>Tests whether a dimension is a positive power of two.</summary>
    /// <param name="value">The dimension length.</param>
    /// <returns>True for a supported radix-2 dimension.</returns>
    private static bool IsPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;
}
