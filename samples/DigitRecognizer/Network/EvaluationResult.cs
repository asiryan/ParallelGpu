namespace DigitRecognizer;

/// <summary>Contains classification measurements for a labeled image selection.</summary>
/// <param name="Count">The number of evaluated images.</param>
/// <param name="MeanLoss">The mean cross-entropy loss.</param>
/// <param name="Accuracy">The fraction of correctly classified images, from zero through one.</param>
public readonly record struct EvaluationResult(int Count, double MeanLoss, double Accuracy);
