using System.Globalization;

namespace RayTracer;

/// <summary>Parses the image, quality, and output settings before GPU initialization.</summary>
internal sealed class RenderSettings
{
    public string OutputPath { get; private set; } = Path.Combine("artifacts", "raytracer.png");
    public int Width { get; private set; } = 960;
    public int Height { get; private set; } = 540;
    public int SamplesPerPixel { get; private set; } = 128;
    public int MaxBounces { get; private set; } = 16;
    public ulong Seed { get; private set; } = 42;
    public bool SaveCheckpoints { get; private set; }
    public bool ShowHelp { get; private set; }

    /// <summary>Accepts an optional PNG path and named rendering options in any order.</summary>
    public static RenderSettings Parse(string[] args)
    {
        var settings = new RenderSettings();
        bool hasOutputPath = false;
        for (int i = 0; i < args.Length; i++)
        {
            string option = args[i];
            switch (option)
            {
                case "--help":
                case "-h":
                    settings.ShowHelp = true;
                    break;
                case "--checkpoints":
                    settings.SaveCheckpoints = true;
                    break;
                case "--width":
                    settings.Width = PositiveInteger(Value(args, ref i), option);
                    break;
                case "--height":
                    settings.Height = PositiveInteger(Value(args, ref i), option);
                    break;
                case "--samples":
                    settings.SamplesPerPixel = PositiveInteger(Value(args, ref i), option);
                    break;
                case "--bounces":
                    settings.MaxBounces = PositiveInteger(Value(args, ref i), option);
                    break;
                case "--seed":
                    if (!ulong.TryParse(Value(args, ref i), NumberStyles.None, CultureInfo.InvariantCulture, out ulong seed))
                        throw new ArgumentException("--seed requires an unsigned 64-bit integer.");
                    settings.Seed = seed;
                    break;
                default:
                    if (option.StartsWith('-') || hasOutputPath)
                        throw new ArgumentException($"Unexpected argument: {option}.");
                    ArgumentException.ThrowIfNullOrWhiteSpace(option);
                    settings.OutputPath = option;
                    hasOutputPath = true;
                    break;
            }
        }

        if (!settings.ShowHelp)
        {
            _ = checked(settings.Width * settings.Height * 3);
            if (!string.Equals(Path.GetExtension(settings.OutputPath), ".png", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("The output path must have a .png extension.");
            settings.OutputPath = Path.GetFullPath(settings.OutputPath);
        }
        return settings;
    }

    private static string Value(string[] args, ref int index)
    {
        if (index + 1 == args.Length)
            throw new ArgumentException($"Missing value for {args[index]}.");
        return args[++index];
    }

    private static int PositiveInteger(string value, string option)
    {
        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int result) || result <= 0)
            throw new ArgumentException($"{option} requires a positive 32-bit integer.");
        return result;
    }
}
