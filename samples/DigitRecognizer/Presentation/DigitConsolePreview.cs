namespace DigitRecognizer;

/// <summary>Displays randomly selected held-out predictions and compact grayscale digits in the console.</summary>
internal static class DigitConsolePreview
{
    /// <summary>Predicts unique held-out examples without changing the model or the training random sequence.</summary>
    /// <param name="network">The trained network holding the labeled dataset on the GPU.</param>
    /// <param name="dataset">The same labeled dataset on the host, used for expected labels and image previews.</param>
    /// <param name="split">The split from which held-out examples are selected.</param>
    /// <param name="count">The maximum number of predictions to display.</param>
    public static void Show(GpuNeuralNetwork network, DigitDataset dataset, DatasetSplit split, int count)
    {
        int[] indices = split.SelectValidationExamples(count);
        if (indices.Length == 0)
            return;
        int[] predictions = network.PredictTrainingRows(indices);
        for (int index = 0; index < indices.Length; index++)
        {
            Console.WriteLine($"Held-out example {index + 1}: expected {dataset.Labels![indices[index]]}, predicted {predictions[index]}");
            PrintDigit(dataset.Pixels, indices[index]);
        }
    }

    /// <summary>Maps normalized grayscale pixels to ASCII shades, averaging pairs of rows for console proportions.</summary>
    /// <param name="pixels">The flat, row-major pixel dataset.</param>
    /// <param name="index">The original image row to display.</param>
    private static void PrintDigit(float[] pixels, int index)
    {
        const string Shades = " .:-=+*#%@";
        int first = index * DigitDataset.InputSize;
        for (int y = 0; y < DigitDataset.ImageWidth; y += 2)
        {
            for (int x = 0; x < DigitDataset.ImageWidth; x++)
            {
                float intensity = (pixels[first + y * DigitDataset.ImageWidth + x] +
                    pixels[first + (y + 1) * DigitDataset.ImageWidth + x]) * 0.5f;
                Console.Write(Shades[(int)(intensity * (Shades.Length - 1))]);
            }
            Console.WriteLine();
        }
    }
}
