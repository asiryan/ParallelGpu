using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Back-propagates gradients through resident layer weights before any parameter update.</summary>
internal static class BackwardOperations
{
    /// <summary>Propagates the following layer's gradients through its original weights and ReLU.</summary>
    /// <param name="gpu">The executor owning all supplied device views.</param>
    /// <param name="batchCount">The actual number of images in the batch.</param>
    /// <param name="activations">The current layer's batch activations.</param>
    /// <param name="nextDeltas">The following layer's pre-activation derivatives.</param>
    /// <param name="nextWeights">The following layer's weights before their update.</param>
    /// <param name="width">The current hidden layer width.</param>
    /// <param name="nextWidth">The following layer width.</param>
    /// <param name="deltas">The current layer's pre-activation derivatives to write.</param>
    internal static void Hidden(Gpu gpu, int batchCount, GpuView<float> activations,
        GpuView<float> nextDeltas, GpuView<float> nextWeights, int width, int nextWidth,
        GpuView<float> deltas)
    {
        gpu.For(0, batchCount * width, index =>
        {
            int row = index / width;
            int neuron = index % width;
            float sum = 0;
            if (activations[index] > 0)
            {
                for (int next = 0; next < nextWidth; next++)
                    sum += nextDeltas[row * nextWidth + next] * nextWeights[neuron * nextWidth + next];
            }
            deltas[index] = sum;
        });
    }
}
