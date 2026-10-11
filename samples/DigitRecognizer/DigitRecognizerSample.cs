using System.Diagnostics;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Coordinates one console run: ZIP loading, GPU training, held-out previews, and output files.</summary>
internal sealed class DigitRecognizerSample
{
    /// <summary>The settings supplied explicitly by Program.cs.</summary>
    private readonly DigitRecognizerSettings settings;

    /// <summary>Creates a sample runner with explicit dataset and training settings.</summary>
    /// <param name="settings">The settings to validate and use for this run.</param>
    /// <exception cref="ArgumentNullException">The settings are null.</exception>
    public DigitRecognizerSample(DigitRecognizerSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        this.settings = settings;
    }

    /// <summary>Loads the dataset and runs training and prediction once, disposing all GPU resources on exit.</summary>
    public void Run()
    {
        settings.Validate();
        var totalTimer = Stopwatch.StartNew();
        var timer = Stopwatch.StartNew();
        SamplePaths paths = SamplePaths.Resolve(settings.DatasetArchive, settings.OutputDirectory);
        DigitArchiveData data = DigitArchive.Load(paths.ArchivePath);
        Console.WriteLine($"Loaded {data.Training.Count:N0} labeled and {data.Test.Count:N0} unlabeled images in {timer.Elapsed.TotalSeconds:F2} s");

        // Split once so the same held-out images remain excluded from every training epoch.
        var split = new DatasetSplit(data.Training.Count, settings.ValidationFraction, settings.Seed);
        // Auto selects CUDA or compatible OpenCL; CPU fallback is disabled.
        using var gpu = new Gpu(new GpuOptions { Backend = GpuBackend.Auto });
        using var network = new GpuNeuralNetwork(gpu, data.Training,
            hiddenLayerSizes: settings.HiddenLayerSizes, batchSize: settings.BatchSize, seed: settings.Seed,
            momentum: settings.Momentum, weightDecay: settings.WeightDecay);
        PrintConfiguration(gpu, split);

        var trainer = new NetworkTrainer(network, settings);
        trainer.Train(split);
        DigitConsolePreview.Show(network, data.Training, split, settings.PredictionExamples);

        // Kaggle test.csv has no labels; infer every row and retain competition ordering in the CSV.
        timer.Restart();
        int[] predictions = network.Predict(data.Test);
        Directory.CreateDirectory(paths.OutputDirectory);
        string submissionPath = Path.Combine(paths.OutputDirectory, "submission.csv");
        KaggleSubmissionWriter.Write(submissionPath, predictions);
        string modelPath = Path.Combine(paths.OutputDirectory, "model.bin");
        network.Save(modelPath);
        totalTimer.Stop();
        Console.WriteLine($"Predicted {predictions.Length:N0} Kaggle images and saved outputs in {timer.Elapsed.TotalSeconds:F2} s");
        Console.WriteLine($"Submission: {submissionPath}");
        Console.WriteLine($"Model: {modelPath}");
        Console.WriteLine($"Total: {totalTimer.Elapsed.TotalSeconds:F2} s");
    }

    /// <summary>Reports the selected GPU, layer widths, dataset split, and optimization settings.</summary>
    /// <param name="gpu">The selected GPU executor.</param>
    /// <param name="split">The training and held-out row indices.</param>
    private void PrintConfiguration(Gpu gpu, DatasetSplit split)
    {
        Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
        Console.WriteLine($"Network: 784 -> {string.Join(" ReLU -> ", settings.HiddenLayerSizes)} ReLU -> 10 Softmax");
        Console.WriteLine($"Training: {split.TrainingIndices.Length:N0}; held-out test: {split.ValidationIndices.Length:N0}; epochs: {settings.Epochs}; batch: {settings.BatchSize}");
        Console.WriteLine($"Learning rate: {settings.LearningRate}; decay per epoch: {settings.LearningRateDecay}; momentum: {settings.Momentum}; weight decay: {settings.WeightDecay}; seed: {settings.Seed}");
    }
}
