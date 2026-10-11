using System.IO.Compression;

namespace DigitRecognizer;

/// <summary>Reads the Kaggle Digit Recognizer CSV files directly from their downloaded ZIP archive.</summary>
public static class DigitArchive
{
    /// <summary>Loads and validates both CSV datasets without extracting the archive to disk.</summary>
    /// <param name="archivePath">The path to the downloaded digit-recognizer.zip file.</param>
    /// <returns>Normalized training pixels with labels, and normalized test pixels without labels.</returns>
    /// <exception cref="ArgumentException">The archive path is null or empty.</exception>
    /// <exception cref="IOException">The archive cannot be opened or read.</exception>
    /// <exception cref="InvalidDataException">The ZIP or a required CSV file is missing, empty, or malformed.</exception>
    public static DigitArchiveData Load(string archivePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(archivePath);
        using var archive = ZipFile.OpenRead(archivePath);
        var trainingEntry = archive.GetEntry("train.csv")
            ?? throw new InvalidDataException("The archive does not contain train.csv.");
        var testEntry = archive.GetEntry("test.csv")
            ?? throw new InvalidDataException("The archive does not contain test.csv.");

        return new(DigitCsvReader.Read(trainingEntry, labeled: true), DigitCsvReader.Read(testEntry, labeled: false));
    }
}
