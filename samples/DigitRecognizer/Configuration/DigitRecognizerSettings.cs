namespace DigitRecognizer;

/// <summary>Contains the dataset, network, training, and preview settings specified by the console entry point.</summary>
internal sealed class DigitRecognizerSettings
{
    /// <summary>Gets the downloaded Kaggle ZIP path, absolute or relative to the data directory.</summary>
    public required string DatasetArchive { get; init; }

    /// <summary>Gets the directory for the trained model and competition predictions.</summary>
    public required string OutputDirectory { get; init; }

    /// <summary>Gets the positive hidden ReLU layer widths in forward-pass order.</summary>
    public required int[] HiddenLayerSizes { get; init; }

    /// <summary>Gets the maximum number of images in one GPU mini-batch.</summary>
    public required int BatchSize { get; init; }

    /// <summary>Gets the number of complete passes over the training split.</summary>
    public required int Epochs { get; init; }

    /// <summary>Gets the initial positive gradient step size.</summary>
    public required float LearningRate { get; init; }

    /// <summary>Gets the multiplier applied to the learning rate after each epoch; one disables decay.</summary>
    public required float LearningRateDecay { get; init; }

    /// <summary>Gets the fraction of the previous parameter update retained by momentum.</summary>
    public required float Momentum { get; init; }

    /// <summary>Gets the nonnegative L2 regularization coefficient for weights.</summary>
    public required float WeightDecay { get; init; }

    /// <summary>Gets the fraction of labeled images reserved for evaluation and excluded from training.</summary>
    public required double ValidationFraction { get; init; }

    /// <summary>Gets the reproducible seed for initialization, dataset splitting, and epoch shuffling.</summary>
    public required int Seed { get; init; }

    /// <summary>Gets the maximum number of randomly chosen held-out images to display after training.</summary>
    public required int PredictionExamples { get; init; }

    /// <summary>Checks configuration before loading the dataset or allocating GPU memory.</summary>
    /// <exception cref="ArgumentException">A path is empty or no hidden layers are specified.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A size, count, fraction, or optimization setting is invalid.</exception>
    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(DatasetArchive);
        ArgumentException.ThrowIfNullOrWhiteSpace(OutputDirectory);
        ArgumentNullException.ThrowIfNull(HiddenLayerSizes);
        if (HiddenLayerSizes.Length == 0)
            throw new ArgumentException("Specify at least one hidden layer.", nameof(HiddenLayerSizes));
        foreach (int width in HiddenLayerSizes)
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(HiddenLayerSizes), "Hidden layer widths must be positive.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(BatchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Epochs);
        ArgumentOutOfRangeException.ThrowIfNegative(PredictionExamples);
        if (!float.IsFinite(LearningRate) || LearningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate));
        if (!float.IsFinite(LearningRateDecay) || LearningRateDecay <= 0 || LearningRateDecay > 1)
            throw new ArgumentOutOfRangeException(nameof(LearningRateDecay));
        if (!float.IsFinite(Momentum) || Momentum < 0 || Momentum >= 1)
            throw new ArgumentOutOfRangeException(nameof(Momentum));
        if (!float.IsFinite(WeightDecay) || WeightDecay < 0)
            throw new ArgumentOutOfRangeException(nameof(WeightDecay));
        if (!double.IsFinite(ValidationFraction) || ValidationFraction <= 0 || ValidationFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(ValidationFraction));
    }
}
