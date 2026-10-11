using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Groups one dense layer's dimensions, parameters, momentum, and batch workspaces.</summary>
/// <remarks>The containing network owns and disposes every buffer; this class only stores their references.</remarks>
internal sealed class GpuNetworkLayer
{
    /// <summary>Gets the number of input features supplied to each neuron.</summary>
    public int InputSize { get; }

    /// <summary>Gets the number of neurons produced by this layer.</summary>
    public int OutputSize { get; }

    /// <summary>Gets the weights indexed by input feature times output width plus neuron.</summary>
    public GpuBuffer<float> Weights { get; }

    /// <summary>Gets one trainable bias per output neuron.</summary>
    public GpuBuffer<float> Bias { get; }

    /// <summary>Gets the previous momentum update for each weight.</summary>
    public GpuBuffer<float> WeightVelocity { get; }

    /// <summary>Gets the previous momentum update for each bias.</summary>
    public GpuBuffer<float> BiasVelocity { get; }

    /// <summary>Gets the row-major batch activations, or linear logits for the output layer.</summary>
    public GpuBuffer<float> Activations { get; }

    /// <summary>Gets the row-major batch derivatives with respect to each neuron's pre-activation.</summary>
    public GpuBuffer<float> Deltas { get; }

    /// <summary>Wraps the validated dimensions and allocated buffers of one dense layer.</summary>
    /// <param name="inputSize">The positive input feature count.</param>
    /// <param name="outputSize">The positive output neuron count.</param>
    /// <param name="weights">The input-by-output weight buffer.</param>
    /// <param name="bias">The output-width bias buffer.</param>
    /// <param name="weightVelocity">The momentum buffer matching the weight buffer.</param>
    /// <param name="biasVelocity">The momentum buffer matching the bias buffer.</param>
    /// <param name="activations">The batch-by-output activation buffer.</param>
    /// <param name="deltas">The derivative buffer matching the activation buffer.</param>
    public GpuNetworkLayer(int inputSize, int outputSize,
        GpuBuffer<float> weights, GpuBuffer<float> bias,
        GpuBuffer<float> weightVelocity, GpuBuffer<float> biasVelocity,
        GpuBuffer<float> activations, GpuBuffer<float> deltas)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = weights;
        Bias = bias;
        WeightVelocity = weightVelocity;
        BiasVelocity = biasVelocity;
        Activations = activations;
        Deltas = deltas;
    }
}
