using ILGPU;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Back-propagates a following layer's gradients through its original weights and ReLU.</summary>
public readonly struct BackwardHiddenKernel : IGpuForKernel
{
    /// <summary>The batch's current layer activations used by the ReLU derivative.</summary>
    private readonly ArrayView<float> activations;
    /// <summary>The following layer's pre-activation derivatives.</summary>
    private readonly ArrayView<float> nextDeltas;
    /// <summary>The following layer's weights before this batch's update.</summary>
    private readonly ArrayView<float> nextWeights;
    /// <summary>The current hidden layer width.</summary>
    private readonly int width;
    /// <summary>The following layer width.</summary>
    private readonly int nextWidth;
    /// <summary>The current layer's pre-activation derivatives to write.</summary>
    private readonly ArrayView<float> deltas;

    /// <summary>Creates a hidden layer's backward pass using the following layer's gradients.</summary>
    /// <param name="activations">The batch's current layer activations.</param>
    /// <param name="nextDeltas">The following layer's pre-activation derivatives.</param>
    /// <param name="nextWeights">The following layer's weights before their update.</param>
    /// <param name="width">The current hidden layer width.</param>
    /// <param name="nextWidth">The following layer width.</param>
    /// <param name="deltas">The current layer's pre-activation derivatives to write.</param>
    public BackwardHiddenKernel(ArrayView<float> activations, ArrayView<float> nextDeltas,
        ArrayView<float> nextWeights, int width, int nextWidth, ArrayView<float> deltas)
    {
        this.activations = activations;
        this.nextDeltas = nextDeltas;
        this.nextWeights = nextWeights;
        this.width = width;
        this.nextWidth = nextWidth;
        this.deltas = deltas;
    }

    /// <summary>Writes the pre-activation derivative for one hidden neuron and image.</summary>
    /// <param name="index">The flattened batch row and hidden neuron index.</param>
    public void Execute(int index)
    {
        int row = index / width;
        int neuron = index % width;
        float sum = 0;
        if (activations[index] > 0)
        {
            // Every hidden layer uses the next layer's weights before any parameter is updated.
            for (int next = 0; next < nextWidth; next++)
                sum += nextDeltas[row * nextWidth + next] * nextWeights[neuron * nextWidth + next];
        }
        deltas[index] = sum;
    }
}

