using System.Globalization;
using System.IO.Compression;

namespace DigitRecognizer;

/// <summary>Parses and validates normalized digit datasets from CSV entries in the downloaded archive.</summary>
internal static class DigitCsvReader
{
    /// <summary>Counts rows first, then parses directly into precisely sized dataset arrays.</summary>
    /// <param name="entry">The CSV member inside the archive.</param>
    /// <param name="labeled">Whether the first column contains a digit label.</param>
    /// <returns>The validated dataset with byte intensities scaled to the range from zero through one.</returns>
    /// <exception cref="InvalidDataException">The header, row width, integer values, or row count is invalid.</exception>
    internal static DigitDataset Read(ZipArchiveEntry entry, bool labeled)
    {
        int count = CountRows(entry, labeled);
        if (count > Array.MaxLength / DigitDataset.InputSize)
            throw new InvalidDataException($"{entry.Name} contains too many images for one pixel array.");

        var pixels = new float[count * DigitDataset.InputSize];
        var labels = labeled ? new int[count] : null;
        using var reader = new StreamReader(entry.Open());
        ValidateHeader(reader.ReadLine(), entry.Name, labeled);

        for (int row = 0; row < count; row++)
        {
            string line = reader.ReadLine()
                ?? throw new InvalidDataException($"{entry.Name} ended before image {row + 1}.");
            ReadOnlySpan<char> remaining = line.AsSpan();
            int column = 0;
            if (labels is not null)
            {
                labels[row] = ReadValue(ref remaining, lastColumn: false, entry.Name, row, column);
                if ((uint)labels[row] >= DigitDataset.ClassCount)
                    throw InvalidCell(entry.Name, row, column, "a digit label from 0 through 9");
                column++;
            }

            int offset = row * DigitDataset.InputSize;
            for (int pixel = 0; pixel < DigitDataset.InputSize; pixel++, column++)
            {
                int value = ReadValue(ref remaining, pixel == DigitDataset.InputSize - 1, entry.Name, row, column);
                if ((uint)value > byte.MaxValue)
                    throw InvalidCell(entry.Name, row, column, "a pixel intensity from 0 through 255");
                pixels[offset + pixel] = value / 255f;
            }
        }

        if (reader.ReadLine() is not null)
            throw new InvalidDataException($"{entry.Name} contains more images than its initial row count.");
        return new(pixels, labels);
    }

    /// <summary>Validates the header and counts image rows before allocating the large pixel array.</summary>
    /// <param name="entry">The CSV member to read.</param>
    /// <param name="labeled">Whether a label column is required.</param>
    /// <returns>The positive number of nonempty image rows.</returns>
    /// <exception cref="InvalidDataException">The file is empty, has an invalid header, or contains a blank row.</exception>
    private static int CountRows(ZipArchiveEntry entry, bool labeled)
    {
        using var reader = new StreamReader(entry.Open());
        ValidateHeader(reader.ReadLine(), entry.Name, labeled);
        int count = 0;
        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
                throw new InvalidDataException($"{entry.Name} contains an empty row at line {count + 2}.");
            if (count == int.MaxValue)
                throw new InvalidDataException($"{entry.Name} contains too many images.");
            count++;
        }

        if (count == 0)
            throw new InvalidDataException($"{entry.Name} does not contain any images.");
        return count;
    }

    /// <summary>Checks the exact Kaggle column names and their order.</summary>
    /// <param name="header">The first CSV line, or null when the entry is empty.</param>
    /// <param name="entryName">The filename used in validation messages.</param>
    /// <param name="labeled">Whether the header begins with label.</param>
    /// <exception cref="InvalidDataException">The header differs from the expected label and pixel columns.</exception>
    private static void ValidateHeader(string? header, string entryName, bool labeled)
    {
        if (header is null)
            throw new InvalidDataException($"{entryName} is empty.");
        ReadOnlySpan<char> remaining = header.AsSpan();
        if (labeled)
        {
            if (!remaining.StartsWith("label,".AsSpan(), StringComparison.Ordinal))
                throw new InvalidDataException($"{entryName} must start with the label column.");
            remaining = remaining[6..];
        }

        for (int pixel = 0; pixel < DigitDataset.InputSize; pixel++)
        {
            int comma = remaining.IndexOf(',');
            bool lastColumn = pixel == DigitDataset.InputSize - 1;
            if ((!lastColumn && comma < 0) || (lastColumn && comma >= 0))
                throw new InvalidDataException($"{entryName} must have exactly {DigitDataset.InputSize} pixel columns.");
            ReadOnlySpan<char> field = comma < 0 ? remaining : remaining[..comma];
            if (!field.SequenceEqual($"pixel{pixel}".AsSpan()))
                throw new InvalidDataException($"{entryName} has an invalid header at pixel{pixel}.");
            remaining = comma < 0 ? [] : remaining[(comma + 1)..];
        }
    }

    /// <summary>Parses one integer cell from a span while enforcing the exact CSV row width.</summary>
    /// <param name="remaining">The unconsumed characters, advanced past the parsed cell.</param>
    /// <param name="lastColumn">Whether this cell must terminate the row.</param>
    /// <param name="entryName">The filename used in validation messages.</param>
    /// <param name="row">The zero-based image row.</param>
    /// <param name="column">The zero-based CSV column.</param>
    /// <returns>The parsed nonnegative decimal integer.</returns>
    /// <exception cref="InvalidDataException">The cell is missing, malformed, or has an unexpected following column.</exception>
    private static int ReadValue(ref ReadOnlySpan<char> remaining, bool lastColumn, string entryName, int row, int column)
    {
        int comma = remaining.IndexOf(',');
        if ((!lastColumn && comma < 0) || (lastColumn && comma >= 0))
            throw InvalidCell(entryName, row, column, "the expected number of CSV columns");
        ReadOnlySpan<char> field = comma < 0 ? remaining : remaining[..comma];
        if (!int.TryParse(field, NumberStyles.None, CultureInfo.InvariantCulture, out int value))
            throw InvalidCell(entryName, row, column, "a nonnegative decimal integer");
        remaining = comma < 0 ? [] : remaining[(comma + 1)..];
        return value;
    }

    /// <summary>Creates a CSV validation error with the source line and column.</summary>
    /// <param name="entryName">The malformed CSV filename.</param>
    /// <param name="row">The zero-based image row, after the header.</param>
    /// <param name="column">The zero-based CSV column.</param>
    /// <param name="expected">The valid cell contents or row shape.</param>
    /// <returns>An exception identifying the invalid cell and its expected format.</returns>
    private static InvalidDataException InvalidCell(string entryName, int row, int column, string expected) =>
        new($"{entryName}, line {row + 2}, column {column + 1}: expected {expected}.");
}
