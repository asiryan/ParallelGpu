using ILGPU;
using ILGPU.Algorithms;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Computes stable softmax probabilities, cross-entropy gradients, and predictions.</summary>
public readonly struct SoftmaxKernel : IGpuForKernel
{
    /// <summary>The batch's row-major output logits.</summary>
    private readonly ArrayView<float> logits;
    /// <summary>The normalized class probabilities to write.</summary>
    private readonly ArrayView<float> probabilities;
    /// <summary>The original image labels, or an empty view for prediction only.</summary>
    private readonly ArrayView<int> labels;
    /// <summary>The selected image indices.</summary>
    private readonly ArrayView<int> order;
    /// <summary>The first selected image in the order buffer.</summary>
    private readonly int sampleOffset;
    /// <summary>The cross-entropy derivatives with respect to output logits.</summary>
    private readonly ArrayView<float> outputDeltas;
    /// <summary>The per-image metrics to reduce after the pass.</summary>
    private readonly ArrayView<NetworkMetric> metrics;
    /// <summary>The first batch row in the metrics buffer.</summary>
    private readonly int metricsOffset;
    /// <summary>The digit predictions to write.</summary>
    private readonly ArrayView<int> predictions;
    /// <summary>The first batch row in the predictions buffer.</summary>
    private readonly int predictionsOffset;

    /// <summary>Creates a probability, gradient, and metric pass for a batch.</summary>
    /// <param name="logits">The ten output logits per image.</param>
    /// <param name="probabilities">The ten probabilities per image to write.</param>
    /// <param name="labels">The original image labels, or an empty view for unlabeled images.</param>
    /// <param name="order">The selected image indices.</param>
    /// <param name="sampleOffset">The first batch image's position in the order buffer.</param>
    /// <param name="outputDeltas">The ten output derivatives per image to write.</param>
    /// <param name="metrics">The per-image loss and correctness values to write.</param>
    /// <param name="metricsOffset">The first batch row in the metrics buffer.</param>
    /// <param name="predictions">The digit predictions to write.</param>
    /// <param name="predictionsOffset">The first batch row in the predictions buffer.</param>
    public SoftmaxKernel(ArrayView<float> logits, ArrayView<float> probabilities,
        ArrayView<int> labels, ArrayView<int> order, int sampleOffset, ArrayView<float> outputDeltas,
        ArrayView<NetworkMetric> metrics, int metricsOffset, ArrayView<int> predictions, int predictionsOffset)
    {
        this.logits = logits;
        this.probabilities = probabilities;
        this.labels = labels;
        this.order = order;
        this.sampleOffset = sampleOffset;
        this.outputDeltas = outputDeltas;
        this.metrics = metrics;
        this.metricsOffset = metricsOffset;
        this.predictions = predictions;
        this.predictionsOffset = predictionsOffset;
    }

    /// <summary>Writes probabilities, derivatives, a prediction, and metrics for one image.</summary>
    /// <param name="index">The zero-based image index within the batch.</param>
    public void Execute(int index)
    {
        int first = index * 10;
        int predicted = 0;
        float maximum = logits[first];
        for (int digit = 1; digit < 10; digit++)
        {
            float value = logits[first + digit];
            if (value > maximum)
            {
                maximum = value;
                predicted = digit;
            }
        }

        // Subtracting the largest logit avoids overflowing exp during training.
        float sum = 0;
        for (int digit = 0; digit < 10; digit++)
        {
            float exponential = XMath.Exp(logits[first + digit] - maximum);
            probabilities[first + digit] = exponential;
            sum += exponential;
        }

        bool labeled = labels.Length > 0;
        int expected = labeled ? labels[order[sampleOffset + index]] : 0;
        for (int digit = 0; digit < 10; digit++)
        {
            float probability = probabilities[first + digit] / sum;
            probabilities[first + digit] = probability;
            outputDeltas[first + digit] = labeled ? probability - (digit == expected ? 1f : 0f) : 0;
        }

        predictions[predictionsOffset + index] = predicted;
        // Log-sum-exp computes cross entropy without taking log of a rounded-to-zero probability.
        metrics[metricsOffset + index] = labeled
            ? new NetworkMetric((maximum - logits[first + expected]) + XMath.Log(sum), predicted == expected ? 1 : 0)
            : default;
    }
}

