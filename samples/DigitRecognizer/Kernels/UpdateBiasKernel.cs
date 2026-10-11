using ILGPU;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Updates one layer's biases from mean batch gradients using momentum.</summary>
public readonly struct UpdateBiasKernel : IGpuForKernel
{
    /// <summary>The layer's row-major pre-activation derivatives.</summary>
    private readonly ArrayView<float> deltas;
    /// <summary>The layer width.</summary>
    private readonly int width;
    /// <summary>The actual number of images in the batch.</summary>
    private readonly int batchCount;
    /// <summary>The biases to update.</summary>
    private readonly ArrayView<float> bias;
    /// <summary>The previous bias update velocities.</summary>
    private readonly ArrayView<float> velocity;
    /// <summary>The gradient step size.</summary>
    private readonly float learningRate;
    /// <summary>The fraction of each previous update retained.</summary>
    private readonly float momentum;

    /// <summary>Creates a bias update for any network layer.</summary>
    /// <param name="deltas">The layer's batch derivatives.</param>
    /// <param name="width">The layer width.</param>
    /// <param name="batchCount">The actual number of images in the batch.</param>
    /// <param name="bias">The biases to update.</param>
    /// <param name="velocity">The bias velocities to update.</param>
    /// <param name="learningRate">The gradient step size.</param>
    /// <param name="momentum">The fraction of the previous update retained.</param>
    public UpdateBiasKernel(ArrayView<float> deltas, int width, int batchCount,
        ArrayView<float> bias, ArrayView<float> velocity, float learningRate, float momentum)
    {
        this.deltas = deltas;
        this.width = width;
        this.batchCount = batchCount;
        this.bias = bias;
        this.velocity = velocity;
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    /// <summary>Updates one bias without applying weight decay.</summary>
    /// <param name="index">The zero-based neuron index within the layer.</param>
    public void Execute(int index)
    {
        float sum = 0;
        for (int row = 0; row < batchCount; row++)
            sum += deltas[row * width + index];
        float update = momentum * velocity[index] - learningRate * sum / batchCount;
        velocity[index] = update;
        bias[index] += update;
    }
}

