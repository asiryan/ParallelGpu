using ILGPU;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Computes the hidden layer and its ReLU activation, one batch neuron per GPU thread.</summary>
public readonly struct ForwardHiddenKernel : IGpuForKernel
{
    /// <summary>The normalized, row-major image pixels.</summary>
    private readonly ArrayView<float> pixels;
    /// <summary>The image indices selected for this pass.</summary>
    private readonly ArrayView<int> order;
    /// <summary>The first selected image in the order buffer.</summary>
    private readonly int sampleOffset;
    /// <summary>The hidden layer width.</summary>
    private readonly int hiddenSize;
    /// <summary>The input-to-hidden weights, indexed by input * hiddenSize + neuron.</summary>
    private readonly ArrayView<float> inputWeights;
    /// <summary>The hidden layer biases.</summary>
    private readonly ArrayView<float> inputBias;
    /// <summary>The row-major hidden activations for the current batch.</summary>
    private readonly ArrayView<float> hidden;

    /// <summary>Creates a forward pass over a selected batch of images.</summary>
    /// <param name="pixels">The normalized image pixels, with 784 values per image.</param>
    /// <param name="order">The selected image indices.</param>
    /// <param name="sampleOffset">The first batch image's position in the order buffer.</param>
    /// <param name="hiddenSize">The hidden layer width.</param>
    /// <param name="inputWeights">The input-to-hidden weights.</param>
    /// <param name="inputBias">The hidden layer biases.</param>
    /// <param name="hidden">The batch's hidden activations to write.</param>
    public ForwardHiddenKernel(ArrayView<float> pixels, ArrayView<int> order, int sampleOffset,
        int hiddenSize, ArrayView<float> inputWeights, ArrayView<float> inputBias, ArrayView<float> hidden)
    {
        this.pixels = pixels;
        this.order = order;
        this.sampleOffset = sampleOffset;
        this.hiddenSize = hiddenSize;
        this.inputWeights = inputWeights;
        this.inputBias = inputBias;
        this.hidden = hidden;
    }

    /// <summary>Writes one hidden neuron's activation for one batch image.</summary>
    /// <param name="index">The flattened batch row and hidden neuron index.</param>
    public void Execute(int index)
    {
        // Each thread owns one image/neuron pair; adjacent neurons read adjacent weights.
        int row = index / hiddenSize;
        int neuron = index % hiddenSize;
        int pixelOffset = order[sampleOffset + row] * 784;
        float sum = inputBias[neuron];
        for (int input = 0; input < 784; input++)
            sum += pixels[pixelOffset + input] * inputWeights[input * hiddenSize + neuron];
        // Preserve NaN so held-out loss detects nonfinite training state instead of silently masking it.
        hidden[index] = sum < 0 ? 0 : sum;
    }
}

