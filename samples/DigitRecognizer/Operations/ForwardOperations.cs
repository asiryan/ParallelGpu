using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Computes input and dense layers with ordinary GPU lambdas over resident buffers.</summary>
internal static class ForwardOperations
{
    /// <summary>Computes the first hidden layer and its ReLU activation, one batch neuron per iteration.</summary>
    /// <param name="gpu">The executor owning all supplied device views.</param>
    /// <param name="batchCount">The actual number of images in the batch.</param>
    /// <param name="pixels">The normalized image pixels, with 784 values per image.</param>
    /// <param name="order">The selected image indices.</param>
    /// <param name="sampleOffset">The first batch image's position in the order buffer.</param>
    /// <param name="hiddenSize">The hidden layer width.</param>
    /// <param name="inputWeights">The input-to-hidden weights.</param>
    /// <param name="inputBias">The hidden layer biases.</param>
    /// <param name="hidden">The batch's hidden activations to write.</param>
    internal static void Hidden(Gpu gpu, int batchCount, GpuView<float> pixels, GpuView<int> order,
        int sampleOffset, int hiddenSize, GpuView<float> inputWeights, GpuView<float> inputBias,
        GpuView<float> hidden)
    {
        gpu.For(0, batchCount * hiddenSize, index =>
        {
            // Each iteration owns one image/neuron pair; adjacent neurons read adjacent weights.
            int row = index / hiddenSize;
            int neuron = index % hiddenSize;
            int pixelOffset = order[sampleOffset + row] * 784;
            float sum = inputBias[neuron];
            for (int input = 0; input < 784; input++)
                sum += pixels[pixelOffset + input] * inputWeights[input * hiddenSize + neuron];
            // Preserve NaN so held-out loss detects nonfinite state instead of silently masking it.
            hidden[index] = sum < 0 ? 0 : sum;
        });
    }

    /// <summary>Computes a dense layer, optionally applying ReLU, one batch neuron per iteration.</summary>
    /// <param name="gpu">The executor owning all supplied device views.</param>
    /// <param name="batchCount">The actual number of images in the batch.</param>
    /// <param name="inputs">The batch's row-major input activations.</param>
    /// <param name="weights">The dense layer weights.</param>
    /// <param name="bias">The dense layer biases.</param>
    /// <param name="inputSize">The input layer width.</param>
    /// <param name="outputSize">The output layer width.</param>
    /// <param name="applyRelu">Whether to apply ReLU after the weighted sum.</param>
    /// <param name="outputs">The batch's output activations or logits to write.</param>
    internal static void Dense(Gpu gpu, int batchCount, GpuView<float> inputs, GpuView<float> weights,
        GpuView<float> bias, int inputSize, int outputSize, bool applyRelu, GpuView<float> outputs)
    {
        gpu.For(0, batchCount * outputSize, index =>
        {
            int row = index / outputSize;
            int neuron = index % outputSize;
            float sum = bias[neuron];
            for (int input = 0; input < inputSize; input++)
                sum += inputs[row * inputSize + input] * weights[input * outputSize + neuron];
            outputs[index] = applyRelu && sum < 0 ? 0 : sum;
        });
    }
}
