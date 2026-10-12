using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Averages batch gradients and updates resident parameters and momentum with GPU lambdas.</summary>
internal static class ParameterUpdateOperations
{
    /// <summary>Updates input weights using momentum and L2 weight decay.</summary>
    /// <param name="gpu">The executor owning all supplied device views.</param>
    /// <param name="pixels">The normalized image pixels.</param>
    /// <param name="order">The selected image indices.</param>
    /// <param name="sampleOffset">The first batch image's position in the order buffer.</param>
    /// <param name="batchCount">The actual number of images, including a shorter final batch.</param>
    /// <param name="hiddenSize">The hidden layer width.</param>
    /// <param name="hiddenDeltas">The batch's hidden pre-activation derivatives.</param>
    /// <param name="inputWeights">The input weights to update.</param>
    /// <param name="inputVelocity">The input weight velocities to update.</param>
    /// <param name="learningRate">The positive gradient step size.</param>
    /// <param name="momentum">The fraction of the previous update retained.</param>
    /// <param name="weightDecay">The L2 regularization coefficient.</param>
    internal static void InputWeights(Gpu gpu, GpuView<float> pixels, GpuView<int> order,
        int sampleOffset, int batchCount, int hiddenSize, GpuView<float> hiddenDeltas,
        GpuView<float> inputWeights, GpuView<float> inputVelocity, float learningRate,
        float momentum, float weightDecay)
    {
        gpu.For(0, inputWeights.Length, index =>
        {
            int input = index / hiddenSize;
            int neuron = index % hiddenSize;
            float sum = 0;
            for (int row = 0; row < batchCount; row++)
                sum += pixels[order[sampleOffset + row] * 784 + input] * hiddenDeltas[row * hiddenSize + neuron];
            float weight = inputWeights[index];
            // A shorter final batch uses the same gradient scale as a full batch.
            float gradient = sum / batchCount + weightDecay * weight;
            float velocity = momentum * inputVelocity[index] - learningRate * gradient;
            inputVelocity[index] = velocity;
            inputWeights[index] = weight + velocity;
        });
    }

    /// <summary>Updates dense layer weights using momentum and L2 weight decay.</summary>
    /// <param name="gpu">The executor owning all supplied device views.</param>
    /// <param name="inputs">The batch's row-major input activations.</param>
    /// <param name="deltas">The batch's output pre-activation derivatives.</param>
    /// <param name="batchCount">The actual number of images, including a shorter final batch.</param>
    /// <param name="inputSize">The input layer width.</param>
    /// <param name="outputSize">The output layer width.</param>
    /// <param name="weights">The dense layer weights to update.</param>
    /// <param name="velocity">The weight velocities to update.</param>
    /// <param name="learningRate">The positive gradient step size.</param>
    /// <param name="momentum">The fraction of the previous update retained.</param>
    /// <param name="weightDecay">The L2 regularization coefficient.</param>
    internal static void DenseWeights(Gpu gpu, GpuView<float> inputs, GpuView<float> deltas,
        int batchCount, int inputSize, int outputSize, GpuView<float> weights,
        GpuView<float> velocity, float learningRate, float momentum, float weightDecay)
    {
        gpu.For(0, weights.Length, index =>
        {
            int input = index / outputSize;
            int output = index % outputSize;
            float sum = 0;
            for (int row = 0; row < batchCount; row++)
                sum += inputs[row * inputSize + input] * deltas[row * outputSize + output];
            float weight = weights[index];
            float gradient = sum / batchCount + weightDecay * weight;
            float update = momentum * velocity[index] - learningRate * gradient;
            velocity[index] = update;
            weights[index] = weight + update;
        });
    }

    /// <summary>Updates biases from mean gradients using momentum without applying weight decay.</summary>
    /// <param name="gpu">The executor owning all supplied device views.</param>
    /// <param name="deltas">The layer's row-major pre-activation derivatives.</param>
    /// <param name="width">The layer width.</param>
    /// <param name="batchCount">The actual number of images, including a shorter final batch.</param>
    /// <param name="bias">The biases to update.</param>
    /// <param name="velocity">The bias velocities to update.</param>
    /// <param name="learningRate">The positive gradient step size.</param>
    /// <param name="momentum">The fraction of the previous update retained.</param>
    internal static void Bias(Gpu gpu, GpuView<float> deltas, int width, int batchCount,
        GpuView<float> bias, GpuView<float> velocity, float learningRate, float momentum)
    {
        gpu.For(0, width, index =>
        {
            float sum = 0;
            for (int row = 0; row < batchCount; row++)
                sum += deltas[row * width + index];
            float update = momentum * velocity[index] - learningRate * sum / batchCount;
            velocity[index] = update;
            bias[index] += update;
        });
    }
}
