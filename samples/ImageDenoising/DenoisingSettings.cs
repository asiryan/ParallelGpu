using System.Globalization;

namespace ImageDenoising;

/// <summary>Parses noise, filter, and file settings before loading images or initializing a GPU.</summary>
internal sealed class DenoisingSettings
{
    public string InputPath { get; private set; } = Path.Combine(AppContext.BaseDirectory, "Assets", "input.png");
    public string OutputDirectory { get; private set; } = Path.Combine("artifacts", "image-denoising");
    public float Sigma { get; private set; } = 20;
    public float Strength { get; private set; } = 0.55f;
    public int SearchRadius { get; private set; } = 10;
    public int PatchRadius { get; private set; } = 1;
    public ulong Seed { get; private set; } = 42;
    public bool SkipNoise { get; private set; }
    public bool ShowHelp { get; private set; }

    /// <summary>Accepts an optional input image followed by named options in any order.</summary>
    public static DenoisingSettings Parse(string[] args)
    {
        var settings = new DenoisingSettings();
        bool hasInputPath = false;
        for (int i = 0; i < args.Length; i++)
        {
            string option = args[i];
            switch (option)
            {
                case "--help":
                case "-h":
                    settings.ShowHelp = true;
                    break;
                case "--skip-noise":
                    settings.SkipNoise = true;
                    break;
                case "--output":
                    settings.OutputDirectory = Value(args, ref i);
                    ArgumentException.ThrowIfNullOrWhiteSpace(settings.OutputDirectory);
                    break;
                case "--sigma":
                    settings.Sigma = Number(Value(args, ref i), option);
                    if (settings.Sigma < 0 || settings.Sigma > 100)
                        throw new ArgumentException("--sigma must be between 0 and 100 in byte intensity units.");
                    break;
                case "--strength":
                    settings.Strength = Number(Value(args, ref i), option);
                    if (settings.Strength <= 0 || settings.Strength > 4)
                        throw new ArgumentException("--strength must be greater than 0 and at most 4.");
                    break;
                case "--search-radius":
                    settings.SearchRadius = Integer(Value(args, ref i), option, 1, 16);
                    break;
                case "--patch-radius":
                    settings.PatchRadius = Integer(Value(args, ref i), option, 0, 4);
                    break;
                case "--seed":
                    if (!ulong.TryParse(Value(args, ref i), NumberStyles.None, CultureInfo.InvariantCulture, out ulong seed))
                        throw new ArgumentException("--seed requires an unsigned 64-bit integer.");
                    settings.Seed = seed;
                    break;
                default:
                    if (option.StartsWith('-') || hasInputPath)
                        throw new ArgumentException($"Unexpected argument: {option}.");
                    ArgumentException.ThrowIfNullOrWhiteSpace(option);
                    settings.InputPath = option;
                    hasInputPath = true;
                    break;
            }
        }

        if (!settings.ShowHelp)
        {
            settings.InputPath = Path.GetFullPath(settings.InputPath);
            settings.OutputDirectory = Path.GetFullPath(settings.OutputDirectory);
            // Prevent output names from overwriting a user-supplied input image.
            foreach (string outputName in new[] { "original.png", "before.png", "after.png" })
                if (string.Equals(settings.InputPath, Path.Combine(settings.OutputDirectory, outputName), StringComparison.OrdinalIgnoreCase))
                    throw new ArgumentException("Choose an output directory that does not overwrite the input image.");
        }
        return settings;
    }

    private static string Value(string[] args, ref int index)
    {
        if (index + 1 == args.Length)
            throw new ArgumentException($"Missing value for {args[index]}.");
        return args[++index];
    }

    private static float Number(string value, string option)
    {
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float number) || !float.IsFinite(number))
            throw new ArgumentException($"{option} requires a finite number, using a dot as the decimal separator.");
        return number;
    }

    private static int Integer(string value, string option, int minimum, int maximum)
    {
        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int number) || number < minimum || number > maximum)
            throw new ArgumentException($"{option} must be an integer between {minimum} and {maximum}.");
        return number;
    }
}
