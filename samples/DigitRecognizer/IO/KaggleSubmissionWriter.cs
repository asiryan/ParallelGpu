using System.Globalization;

namespace DigitRecognizer;

/// <summary>Writes the Digit Recognizer competition's ImageId,Label prediction file.</summary>
internal static class KaggleSubmissionWriter
{
    /// <summary>Writes one-based image IDs and predicted digit labels in the original test.csv order.</summary>
    /// <param name="path">The destination CSV path; its parent directory must exist.</param>
    /// <param name="predictions">One predicted digit per test image, in dataset order.</param>
    /// <exception cref="IOException">The CSV cannot be written.</exception>
    public static void Write(string path, int[] predictions)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("ImageId,Label");
        for (int index = 0; index < predictions.Length; index++)
            writer.WriteLine($"{(index + 1).ToString(CultureInfo.InvariantCulture)},{predictions[index].ToString(CultureInfo.InvariantCulture)}");
    }
}
