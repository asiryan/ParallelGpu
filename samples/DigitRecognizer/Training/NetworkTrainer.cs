using System.Diagnostics;

namespace DigitRecognizer;

/// <summary>Runs kernel warm-up, complete training epochs, and held-out evaluation with console progress.</summary>
internal sealed class NetworkTrainer
{
    /// <summary>The borrowed network whose parameters this trainer updates.</summary>
    private readonly GpuNeuralNetwork network;

    /// <summary>The epoch count and optimization settings supplied by the entry point.</summary>
    private readonly DigitRecognizerSettings settings;

    /// <summary>Creates a trainer for an initialized network.</summary>
    /// <param name="network">The network owned by the sample runner.</param>
    /// <param name="settings">The validated training settings.</param>
    public NetworkTrainer(GpuNeuralNetwork network, DigitRecognizerSettings settings)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(settings);
        this.network = network;
        this.settings = settings;
    }

    /// <summary>Trains on every selected training row each epoch and evaluates the same held-out set.</summary>
    /// <param name="split">The separate training and validation row indices.</param>
    /// <returns>The final held-out loss and accuracy after all epochs.</returns>
    public EvaluationResult Train(DatasetSplit split)
    {
        ArgumentNullException.ThrowIfNull(split);
        var timer = Stopwatch.StartNew();
        network.WarmUp();
        Console.WriteLine($"Kernel warm-up: {timer.Elapsed.TotalSeconds:F2} s");
        EvaluationResult initial = network.Evaluate(split.ValidationIndices);
        Console.WriteLine($"Before training: held-out loss {initial.MeanLoss:F4}, accuracy {initial.Accuracy:P2}");

        var trainingTimer = Stopwatch.StartNew();
        EvaluationResult final = initial;
        for (int epochIndex = 0; epochIndex < settings.Epochs; epochIndex++)
        {
            split.ShuffleTraining();
            // Smaller steps in later epochs let the deeper network refine its learned weights.
            float learningRate = settings.LearningRate * MathF.Pow(settings.LearningRateDecay, epochIndex);
            timer.Restart();
            network.TrainEpoch(split.TrainingIndices, learningRate);
            double trainingSeconds = timer.Elapsed.TotalSeconds;
            final = network.Evaluate(split.ValidationIndices);
            Console.WriteLine($"Epoch {epochIndex + 1}/{settings.Epochs}: learning rate {learningRate:F4}, held-out loss {final.MeanLoss:F4}, accuracy {final.Accuracy:P2}, training {trainingSeconds:F2} s, including evaluation {timer.Elapsed.TotalSeconds:F2} s");
        }
        trainingTimer.Stop();
        Console.WriteLine($"Final held-out test: {final.Accuracy:P2} accuracy on {final.Count:N0} images; loss {final.MeanLoss:F4}");
        Console.WriteLine($"Training and evaluation: {trainingTimer.Elapsed.TotalSeconds:F2} s");
        return final;
    }
}
