using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Writes layer dimensions and learned parameters in the sample's PGN2 binary model format.</summary>
internal static class NetworkModelWriter
{
    /// <summary>The format marker identifying the version that supports multiple dense layers.</summary>
    private const int FormatMarker = 0x50474E32;

    /// <summary>Saves dimensions followed by each layer's weights and biases in forward-pass order.</summary>
    /// <param name="path">The destination filename; its parent directory must exist.</param>
    /// <param name="layers">The live network layers in forward-pass order.</param>
    /// <remarks>Momentum and batch workspaces are omitted; saving downloads only learned parameters.</remarks>
    /// <exception cref="ArgumentException">The destination path is invalid.</exception>
    /// <exception cref="IOException">The destination cannot be created or written.</exception>
    /// <exception cref="UnauthorizedAccessException">The destination cannot be written by the current process.</exception>
    /// <exception cref="ObjectDisposedException">A layer's parameter buffer or its executor has been disposed.</exception>
    public static void Write(string path, IReadOnlyList<GpuNetworkLayer> layers)
    {
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(FormatMarker);
        writer.Write(layers.Count + 1);
        writer.Write(DigitDataset.InputSize);
        foreach (var layer in layers)
            writer.Write(layer.OutputSize);
        foreach (var layer in layers)
        {
            WriteValues(writer, layer.Weights);
            WriteValues(writer, layer.Bias);
        }
    }

    /// <summary>Downloads one parameter buffer and writes its floats in their existing order.</summary>
    /// <param name="writer">The open model writer, retained by the caller.</param>
    /// <param name="buffer">The live weight or bias buffer to serialize.</param>
    private static void WriteValues(BinaryWriter writer, GpuBuffer<float> buffer)
    {
        foreach (float value in buffer.ToArray())
            writer.Write(value);
    }
}
