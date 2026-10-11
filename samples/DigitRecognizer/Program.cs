using System.Diagnostics;
using System.Globalization;
using DigitRecognizer;
using ParallelGpu;

// Change dataset paths and training settings here. Relative paths also work when launched from the IDE.
// https://www.kaggle.com/competitions/digit-recognizer/data
const string DatasetArchive = ".artifacts/digit-recognizer.zip";
const string OutputDirectory = ".artifacts/digit-recognizer";
const int HiddenSize = 128;
const int BatchSize = 256;
const int Epochs = 10;
const float LearningRate = 0.05f;
const float Momentum = 0.9f;
const float WeightDecay = 0.0001f;
const double ValidationFraction = 0.1;
const int Seed = 12345;
const int PredictionExamples = 5;

try
{
    ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Epochs);
    if (!double.IsFinite(ValidationFraction) || ValidationFraction <= 0 || ValidationFraction >= 1)
        throw new ArgumentOutOfRangeException(nameof(ValidationFraction));
    ArgumentOutOfRangeException.ThrowIfNegative(PredictionExamples);
    if (!float.IsFinite(LearningRate) || LearningRate <= 0)
        throw new ArgumentOutOfRangeException(nameof(LearningRate));
    var totalTimer = Stopwatch.StartNew();
    var timer = Stopwatch.StartNew();
    string dataDirectory = FindDataDirectory(DatasetArchive);
    DigitArchiveData data = DigitArchive.Load(Path.GetFullPath(DatasetArchive, dataDirectory));
    Console.WriteLine($"Loaded {data.Training.Count:N0} labeled and {data.Test.Count:N0} unlabeled images in {timer.Elapsed.TotalSeconds:F2} s");

    // Split once. Held-out images never participate in the epoch's weight updates.
    var random = new Random(Seed);
    int[] allIndices = Enumerable.Range(0, data.Training.Count).ToArray();
    Shuffle(allIndices, random);
    int validationCount = (int)(data.Training.Count * ValidationFraction);
    if (validationCount < 1 || validationCount >= data.Training.Count)
        throw new ArgumentException("The validation split must leave images in both training and validation sets.");
    int[] validationIndices = allIndices[..validationCount];
    int[] trainingIndices = allIndices[validationCount..];

    // Auto selects CUDA or compatible OpenCL; CPU fallback is disabled.
    using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
    using var network = new GpuNeuralNetwork(gpu, data.Training,
        hiddenSize: HiddenSize, batchSize: BatchSize, seed: Seed,
        momentum: Momentum, weightDecay: WeightDecay);
    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Network: 784 -> {HiddenSize} ReLU -> 10 Softmax");
    Console.WriteLine($"Training: {trainingIndices.Length:N0}; held-out test: {validationIndices.Length:N0}; epochs: {Epochs}; batch: {BatchSize}");
    Console.WriteLine($"Learning rate: {LearningRate}; momentum: {Momentum}; weight decay: {WeightDecay}; seed: {Seed}");

    timer.Restart();
    network.WarmUp();
    Console.WriteLine($"Kernel warm-up: {timer.Elapsed.TotalSeconds:F2} s");
    EvaluationResult initial = network.Evaluate(validationIndices);
    Console.WriteLine($"Before training: held-out loss {initial.MeanLoss:F4}, accuracy {initial.Accuracy:P2}");

    var trainingTimer = Stopwatch.StartNew();
    EvaluationResult final = initial;
    for (int epoch = 1; epoch <= Epochs; epoch++)
    {
        Shuffle(trainingIndices, random);
        timer.Restart();
        network.TrainEpoch(trainingIndices, LearningRate);
        double trainingSeconds = timer.Elapsed.TotalSeconds;
        final = network.Evaluate(validationIndices);
        Console.WriteLine($"Epoch {epoch}/{Epochs}: held-out loss {final.MeanLoss:F4}, accuracy {final.Accuracy:P2}, training {trainingSeconds:F2} s, including evaluation {timer.Elapsed.TotalSeconds:F2} s");
    }
    trainingTimer.Stop();
    Console.WriteLine($"Final held-out test: {final.Accuracy:P2} accuracy on {final.Count:N0} images; loss {final.MeanLoss:F4}");
    Console.WriteLine($"Training and evaluation: {trainingTimer.Elapsed.TotalSeconds:F2} s");

    // Vary console previews without changing the reproducible training or held-out split.
    int[] examplePool = validationIndices.ToArray();
    Shuffle(examplePool, Random.Shared);
    int[] exampleIndices = examplePool[..Math.Min(PredictionExamples, examplePool.Length)];
    if (exampleIndices.Length > 0)
    {
        int[] examples = network.PredictTrainingRows(exampleIndices);
        for (int i = 0; i < exampleIndices.Length; i++)
        {
            Console.WriteLine($"Held-out example {i + 1}: expected {data.Training.Labels![exampleIndices[i]]}, predicted {examples[i]}");
            PrintDigit(data.Training.Pixels, exampleIndices[i]);
        }
    }

    // Kaggle test.csv has no labels. Run inference and write its predictions in competition order.
    timer.Restart();
    int[] predictions = network.Predict(data.Test);
    string outputDirectory = Path.GetFullPath(OutputDirectory, dataDirectory);
    Directory.CreateDirectory(outputDirectory);
    string submissionPath = Path.Combine(outputDirectory, "submission.csv");
    using (var writer = new StreamWriter(submissionPath))
    {
        writer.WriteLine("ImageId,Label");
        for (int i = 0; i < predictions.Length; i++)
            writer.WriteLine($"{(i + 1).ToString(CultureInfo.InvariantCulture)},{predictions[i].ToString(CultureInfo.InvariantCulture)}");
    }
    string modelPath = Path.Combine(outputDirectory, "model.bin");
    network.Save(modelPath);
    totalTimer.Stop();
    Console.WriteLine($"Predicted {predictions.Length:N0} Kaggle images and saved outputs in {timer.Elapsed.TotalSeconds:F2} s");
    Console.WriteLine($"Submission: {Path.GetFullPath(submissionPath)}");
    Console.WriteLine($"Model: {Path.GetFullPath(modelPath)}");
    Console.WriteLine($"Total: {totalTimer.Elapsed.TotalSeconds:F2} s");
    return 0;
}
catch (Exception error) when (error is ArgumentException or IOException or InvalidDataException or
    UnauthorizedAccessException or InvalidOperationException or OverflowException)
{
    Console.Error.WriteLine($"DigitRecognizer failed: {error.Message}");
    return 1;
}

// Prefer the current directory, then locate the archive above the executable for IDE launches.
static string FindDataDirectory(string archivePath)
{
    if (File.Exists(archivePath))
        return Environment.CurrentDirectory;
    for (DirectoryInfo? directory = new(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        if (File.Exists(Path.Combine(directory.FullName, archivePath)))
            return directory.FullName;
    throw new FileNotFoundException("Set DatasetArchive to the downloaded Digit Recognizer ZIP.", archivePath);
}

// Fisher-Yates reshuffles only row indices; the large pixel dataset stays on the GPU.
static void Shuffle(int[] values, Random random)
{
    for (int i = values.Length - 1; i > 0; i--)
    {
        int other = random.Next(i + 1);
        (values[i], values[other]) = (values[other], values[i]);
    }
}

// A compact console preview is sufficient to inspect predictions without a UI.
static void PrintDigit(float[] pixels, int index)
{
    const string Shades = " .:-=+*#%@";
    int first = index * DigitDataset.InputSize;
    for (int y = 0; y < DigitDataset.ImageWidth; y += 2)
    {
        for (int x = 0; x < DigitDataset.ImageWidth; x++)
        {
            float intensity = (pixels[first + y * DigitDataset.ImageWidth + x] +
                pixels[first + (y + 1) * DigitDataset.ImageWidth + x]) * 0.5f;
            Console.Write(Shades[(int)(intensity * (Shades.Length - 1))]);
        }
        Console.WriteLine();
    }
}
