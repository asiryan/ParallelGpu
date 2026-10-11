using ILGPU;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Averages batch gradients and updates dense layer weights using momentum and L2 decay.</summary>
public readonly struct UpdateDenseWeightsKernel : IGpuForKernel
{
    /// <summary>The batch's input activations.</summary>
    private readonly ArrayView<float> inputs;
    /// <summary>The batch's output pre-activation derivatives.</summary>
    private readonly ArrayView<float> deltas;
    /// <summary>The actual number of images in the batch.</summary>
    private readonly int batchCount;
    /// <summary>The input layer width.</summary>
    private readonly int inputSize;
    /// <summary>The output layer width.</summary>
    private readonly int outputSize;
    /// <summary>The dense layer weights to update.</summary>
    private readonly ArrayView<float> weights;
    /// <summary>The previous weight update velocities.</summary>
    private readonly ArrayView<float> velocity;
    /// <summary>The gradient step size.</summary>
    private readonly float learningRate;
    /// <summary>The fraction of each previous update retained.</summary>
    private readonly float momentum;
    /// <summary>The L2 coefficient applied to weights.</summary>
    private readonly float weightDecay;

    /// <summary>Creates a dense layer weight update for a completed backward pass.</summary>
    /// <param name="inputs">The batch's input activations.</param>
    /// <param name="deltas">The batch's output pre-activation derivatives.</param>
    /// <param name="batchCount">The actual number of images in the batch.</param>
    /// <param name="inputSize">The input layer width.</param>
    /// <param name="outputSize">The output layer width.</param>
    /// <param name="weights">The dense layer weights to update.</param>
    /// <param name="velocity">The weight velocities to update.</param>
    /// <param name="learningRate">The gradient step size.</param>
    /// <param name="momentum">The fraction of the previous update retained.</param>
    /// <param name="weightDecay">The L2 regularization coefficient.</param>
    public UpdateDenseWeightsKernel(ArrayView<float> inputs, ArrayView<float> deltas,
        int batchCount, int inputSize, int outputSize, ArrayView<float> weights, ArrayView<float> velocity,
        float learningRate, float momentum, float weightDecay)
    {
        this.inputs = inputs;
        this.deltas = deltas;
        this.batchCount = batchCount;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = weights;
        this.velocity = velocity;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
    }

    /// <summary>Updates one dense layer weight and its momentum velocity.</summary>
    /// <param name="index">The flattened input and output neuron index.</param>
    public void Execute(int index)
    {
        int input = index / outputSize;
        int output = index % outputSize;
        // A weight's gradient is the sum of input * output delta over all images in this batch.
        float sum = 0;
        for (int row = 0; row < batchCount; row++)
            sum += inputs[row * inputSize + input] * deltas[row * outputSize + output];
        float weight = weights[index];
        float gradient = sum / batchCount + weightDecay * weight;
        float update = momentum * velocity[index] - learningRate * gradient;
        velocity[index] = update;
        weights[index] = weight + update;
    }
}

