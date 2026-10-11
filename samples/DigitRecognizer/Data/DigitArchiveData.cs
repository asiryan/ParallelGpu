namespace DigitRecognizer;

/// <summary>Contains the labeled training set and unlabeled competition test set loaded from one archive.</summary>
/// <param name="Training">The labeled images used for training and local validation.</param>
/// <param name="Test">The unlabeled images used to produce competition predictions.</param>
public readonly record struct DigitArchiveData(DigitDataset Training, DigitDataset Test);
