using DigitRecognizer;

// Change dataset paths, layer widths, and training settings here.
// Dataset: https://www.kaggle.com/competitions/digit-recognizer/data
var settings = new DigitRecognizerSettings
{
    DatasetArchive = ".artifacts/digit-recognizer.zip",
    OutputDirectory = ".artifacts/digit-recognizer",
    HiddenLayerSizes = [512, 256],
    BatchSize = 256,
    Epochs = 14,
    LearningRate = 0.05f,
    LearningRateDecay = 0.95f,
    Momentum = 0.9f,
    WeightDecay = 0.0001f,
    ValidationFraction = 0.1,
    Seed = 12345,
    PredictionExamples = 5
};

try
{
    new DigitRecognizerSample(settings).Run();
    return 0;
}
catch (Exception error) when (error is ArgumentException or IOException or InvalidDataException or
    UnauthorizedAccessException or InvalidOperationException or OverflowException)
{
    Console.Error.WriteLine($"DigitRecognizer failed: {error.Message}");
    return 1;
}
