using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Computes stable probabilities, cross-entropy gradients, and predictions on the GPU.</summary>
internal static class SoftmaxOperations
{
    /// <summary>Writes probabilities, derivatives, a prediction, and metrics for each batch image.</summary>
    /// <param name="gpu">The executor owning all supplied device views.</param>
    /// <param name="batchCount">The actual number of images in the batch.</param>
    /// <param name="logits">The ten output logits per image.</param>
    /// <param name="probabilities">The ten probabilities per image to write.</param>
    /// <param name="labels">The original labels, or an empty view for unlabeled prediction.</param>
    /// <param name="order">The selected image indices.</param>
    /// <param name="sampleOffset">The first batch image's position in the order buffer.</param>
    /// <param name="outputDeltas">The ten output derivatives per image to write.</param>
    /// <param name="metrics">The per-image loss and correctness values to write.</param>
    /// <param name="metricsOffset">The first batch row in the metrics buffer.</param>
    /// <param name="predictions">The digit predictions to write.</param>
    /// <param name="predictionsOffset">The first batch row in the predictions buffer.</param>
    internal static void Compute(Gpu gpu, int batchCount, GpuView<float> logits,
        GpuView<float> probabilities, GpuView<int> labels, GpuView<int> order, int sampleOffset,
        GpuView<float> outputDeltas, GpuView<NetworkMetric> metrics, int metricsOffset,
        GpuView<int> predictions, int predictionsOffset)
    {
        gpu.For(0, batchCount, index =>
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
                float exponential = MathF.Exp(logits[first + digit] - maximum);
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
            // Log-sum-exp avoids taking log of a rounded-to-zero probability.
            metrics[metricsOffset + index] = labeled
                ? new NetworkMetric((maximum - logits[first + expected]) + MathF.Log(sum), predicted == expected ? 1 : 0)
                : default;
        });
    }
}
