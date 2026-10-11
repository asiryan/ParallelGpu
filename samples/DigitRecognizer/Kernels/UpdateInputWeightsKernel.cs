using ILGPU;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Averages batch gradients and updates input weights using momentum and L2 weight decay.</summary>
public readonly struct UpdateInputWeightsKernel : IGpuForKernel
{
    /// <summary>The normalized image pixels.</summary>
    private readonly ArrayView<float> pixels;
    /// <summary>The selected image indices.</summary>
    private readonly ArrayView<int> order;
    /// <summary>The first selected image in the order buffer.</summary>
    private readonly int sampleOffset;
    /// <summary>The actual batch size, including a final incomplete batch.</summary>
    private readonly int batchCount;
    /// <summary>The hidden layer width.</summary>
    private readonly int hiddenSize;
    /// <summary>The batch's hidden pre-activation derivatives.</summary>
    private readonly ArrayView<float> hiddenDeltas;
    /// <summary>The input weights to update.</summary>
    private readonly ArrayView<float> inputWeights;
    /// <summary>The previous input weight update velocities.</summary>
    private readonly ArrayView<float> inputVelocity;
    /// <summary>The gradient step size.</summary>
    private readonly float learningRate;
    /// <summary>The fraction of each previous update retained.</summary>
    private readonly float momentum;
    /// <summary>The L2 coefficient applied to weights.</summary>
    private readonly float weightDecay;

    /// <summary>Creates an input weight update for a completed backward pass.</summary>
    /// <param name="pixels">The normalized image pixels.</param>
    /// <param name="order">The selected image indices.</param>
    /// <param name="sampleOffset">The first batch image's position in the order buffer.</param>
    /// <param name="batchCount">The actual number of images in the batch.</param>
    /// <param name="hiddenSize">The hidden layer width.</param>
    /// <param name="hiddenDeltas">The batch's hidden derivatives.</param>
    /// <param name="inputWeights">The input weights to update.</param>
    /// <param name="inputVelocity">The input weight velocities to update.</param>
    /// <param name="learningRate">The gradient step size.</param>
    /// <param name="momentum">The fraction of the previous update retained.</param>
    /// <param name="weightDecay">The L2 regularization coefficient.</param>
    public UpdateInputWeightsKernel(ArrayView<float> pixels, ArrayView<int> order, int sampleOffset,
        int batchCount, int hiddenSize, ArrayView<float> hiddenDeltas, ArrayView<float> inputWeights,
        ArrayView<float> inputVelocity, float learningRate, float momentum, float weightDecay)
    {
        this.pixels = pixels;
        this.order = order;
        this.sampleOffset = sampleOffset;
        this.batchCount = batchCount;
        this.hiddenSize = hiddenSize;
        this.hiddenDeltas = hiddenDeltas;
        this.inputWeights = inputWeights;
        this.inputVelocity = inputVelocity;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
    }

    /// <summary>Updates one input weight and its momentum velocity.</summary>
    /// <param name="index">The flattened input pixel and hidden neuron index.</param>
    public void Execute(int index)
    {
        int input = index / hiddenSize;
        int neuron = index % hiddenSize;
        float sum = 0;
        for (int row = 0; row < batchCount; row++)
            sum += pixels[order[sampleOffset + row] * 784 + input] * hiddenDeltas[row * hiddenSize + neuron];
        float weight = inputWeights[index];
        // Average by the actual batch size so a shorter final batch has the same gradient scale.
        float gradient = sum / batchCount + weightDecay * weight;
        float velocity = momentum * inputVelocity[index] - learningRate * gradient;
        inputVelocity[index] = velocity;
        inputWeights[index] = weight + velocity;
    }
}

