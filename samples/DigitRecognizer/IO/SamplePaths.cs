namespace DigitRecognizer;

/// <summary>Contains absolute paths resolved for a repository-root or IDE launch.</summary>
/// <param name="ArchivePath">The downloaded dataset ZIP.</param>
/// <param name="OutputDirectory">The destination directory for the model and submission.</param>
internal readonly record struct SamplePaths(string ArchivePath, string OutputDirectory)
{
    /// <summary>Resolves both settings relative to the directory from which the configured archive can be found.</summary>
    /// <param name="archivePath">The configured absolute or relative dataset ZIP path.</param>
    /// <param name="outputDirectory">The configured absolute or relative result directory.</param>
    /// <returns>The absolute dataset and output paths.</returns>
    /// <exception cref="FileNotFoundException">The archive cannot be found from the working or executable directories.</exception>
    public static SamplePaths Resolve(string archivePath, string outputDirectory)
    {
        string dataDirectory = FindDataDirectory(archivePath);
        return new(Path.GetFullPath(archivePath, dataDirectory), Path.GetFullPath(outputDirectory, dataDirectory));
    }

    /// <summary>Prefers the working directory, then searches above the executable for IDE launches.</summary>
    /// <param name="archivePath">The configured dataset ZIP path.</param>
    /// <returns>The base directory from which that configured path resolves to an existing file.</returns>
    /// <exception cref="FileNotFoundException">The configured archive is missing.</exception>
    private static string FindDataDirectory(string archivePath)
    {
        if (File.Exists(archivePath))
            return Environment.CurrentDirectory;
        for (DirectoryInfo? directory = new(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, archivePath)))
                return directory.FullName;
        throw new FileNotFoundException("Set DatasetArchive to the downloaded Digit Recognizer ZIP.", archivePath);
    }
}
