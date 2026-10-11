namespace DigitRecognizer;

/// <summary>Separates labeled rows into reproducible training and held-out sets without copying image pixels.</summary>
internal sealed class DatasetSplit
{
    /// <summary>The seeded generator shared by the initial split and subsequent training shuffles.</summary>
    private readonly Random random;

    /// <summary>Gets the training row indices, whose order is reshuffled before every epoch.</summary>
    public int[] TrainingIndices { get; }

    /// <summary>Gets the held-out row indices, which never participate in parameter updates.</summary>
    public int[] ValidationIndices { get; }

    /// <summary>Creates one shuffled split of the original labeled row indices.</summary>
    /// <param name="imageCount">The number of labeled images.</param>
    /// <param name="validationFraction">The fraction reserved for held-out evaluation.</param>
    /// <param name="seed">The seed for the split and all later training shuffles.</param>
    /// <exception cref="ArgumentOutOfRangeException">The image count or validation fraction is invalid.</exception>
    /// <exception cref="ArgumentException">The split would leave either set empty.</exception>
    public DatasetSplit(int imageCount, double validationFraction, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(imageCount);
        if (!double.IsFinite(validationFraction) || validationFraction <= 0 || validationFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(validationFraction));
        int validationCount = (int)(imageCount * validationFraction);
        if (validationCount < 1 || validationCount >= imageCount)
            throw new ArgumentException("The validation split must leave images in both training and validation sets.");

        random = new Random(seed);
        int[] indices = Enumerable.Range(0, imageCount).ToArray();
        Shuffle(indices, random);
        ValidationIndices = indices[..validationCount];
        TrainingIndices = indices[validationCount..];
    }

    /// <summary>Reshuffles only training row indices; the uploaded dataset stays in place on the GPU.</summary>
    public void ShuffleTraining() => Shuffle(TrainingIndices, random);

    /// <summary>Selects unique held-out previews using randomness independent of the seeded training sequence.</summary>
    /// <param name="count">The maximum number of examples to select; zero disables previews.</param>
    /// <returns>The selected original row indices, capped by the held-out image count.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The requested count is negative.</exception>
    public int[] SelectValidationExamples(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (count == 0)
            return [];
        int[] examples = ValidationIndices.ToArray();
        Shuffle(examples, Random.Shared);
        return examples[..Math.Min(count, examples.Length)];
    }

    /// <summary>Uses Fisher-Yates to shuffle an index array without selecting any row twice.</summary>
    /// <param name="values">The row indices to reorder in place.</param>
    /// <param name="generator">The generator for unbiased swap positions.</param>
    private static void Shuffle(int[] values, Random generator)
    {
        for (int index = values.Length - 1; index > 0; index--)
        {
            int other = generator.Next(index + 1);
            (values[index], values[other]) = (values[other], values[index]);
        }
    }
}
