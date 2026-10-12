namespace DigitRecognizer;

/// <summary>Contains an image's cross-entropy loss and whether its prediction was correct.</summary>
public struct NetworkMetric
{
    /// <summary>The cross-entropy loss, summed when this value represents multiple images.</summary>
    public double Loss;
    /// <summary>The number of correctly classified images.</summary>
    public int Correct;

    /// <summary>Creates a device-compatible loss and correctness value.</summary>
    /// <param name="loss">The cross-entropy loss.</param>
    /// <param name="correct">The number of correct predictions.</param>
    public NetworkMetric(double loss, int correct)
    {
        Loss = loss;
        Correct = correct;
    }
}

