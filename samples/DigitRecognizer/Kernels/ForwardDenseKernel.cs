using ILGPU;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Computes a dense layer, optionally applying ReLU, one batch neuron per GPU thread.</summary>
public readonly struct ForwardDenseKernel : IGpuForKernel
{
    /// <summary>The batch's row-major input activations.</summary>
    private readonly ArrayView<float> inputs;
    /// <summary>The layer weights, indexed by input * outputSize + neuron.</summary>
    private readonly ArrayView<float> weights;
    /// <summary>The layer biases.</summary>
    private readonly ArrayView<float> bias;
    /// <summary>The input layer width.</summary>
    private readonly int inputSize;
    /// <summary>The output layer width.</summary>
    private readonly int outputSize;
    /// <summary>The native integer flag enabling ReLU for a hidden layer.</summary>
    private readonly int applyRelu;
    /// <summary>The row-major activations or logits to write.</summary>
    private readonly ArrayView<float> outputs;

    /// <summary>Creates a hidden or output dense layer's forward pass.</summary>
    /// <param name="inputs">The batch's input activations.</param>
    /// <param name="weights">The dense layer's weights.</param>
    /// <param name="bias">The dense layer's biases.</param>
    /// <param name="inputSize">The input layer width.</param>
    /// <param name="outputSize">The output layer width.</param>
    /// <param name="applyRelu">Whether to apply ReLU after the weighted sum.</param>
    /// <param name="outputs">The batch's output activations or logits to write.</param>
    public ForwardDenseKernel(ArrayView<float> inputs, ArrayView<float> weights,
        ArrayView<float> bias, int inputSize, int outputSize, bool applyRelu, ArrayView<float> outputs)
    {
        this.inputs = inputs;
        this.weights = weights;
        this.bias = bias;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        // Native kernel state stores Boolean flags as integers to remain blittable.
        this.applyRelu = applyRelu ? 1 : 0;
        this.outputs = outputs;
    }

    /// <summary>Writes one neuron's activation or logit for one image.</summary>
    /// <param name="index">The flattened batch row and output neuron index.</param>
    public void Execute(int index)
    {
        // Batch rows and layer widths determine the same flattened layout for every dense layer.
        int row = index / outputSize;
        int neuron = index % outputSize;
        float sum = bias[neuron];
        for (int input = 0; input < inputSize; input++)
            sum += inputs[row * inputSize + input] * weights[input * outputSize + neuron];
        outputs[index] = applyRelu != 0 && sum < 0 ? 0 : sum;
    }
}

