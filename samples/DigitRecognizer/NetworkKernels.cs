using ILGPU;
using ILGPU.Algorithms;
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
        int row = index / hiddenSize;
        int neuron = index % hiddenSize;
        int pixelOffset = order[sampleOffset + row] * 784;
        float sum = inputBias[neuron];
        for (int input = 0; input < 784; input++)
            sum += pixels[pixelOffset + input] * inputWeights[input * hiddenSize + neuron];
        hidden[index] = sum > 0 ? sum : 0;
    }
}

/// <summary>Computes the ten output logits, one batch class per GPU thread.</summary>
public readonly struct ForwardOutputKernel : IGpuForKernel
{
    /// <summary>The batch's row-major hidden activations.</summary>
    private readonly ArrayView<float> hidden;
    /// <summary>The hidden-to-output weights, indexed by neuron * 10 + digit.</summary>
    private readonly ArrayView<float> outputWeights;
    /// <summary>The output layer biases.</summary>
    private readonly ArrayView<float> outputBias;
    /// <summary>The hidden layer width.</summary>
    private readonly int hiddenSize;
    /// <summary>The row-major output logits to write.</summary>
    private readonly ArrayView<float> logits;

    /// <summary>Creates an output layer forward pass.</summary>
    /// <param name="hidden">The batch's hidden activations.</param>
    /// <param name="outputWeights">The hidden-to-output weights.</param>
    /// <param name="outputBias">The output biases.</param>
    /// <param name="hiddenSize">The hidden layer width.</param>
    /// <param name="logits">The batch's ten output logits per image to write.</param>
    public ForwardOutputKernel(ArrayView<float> hidden, ArrayView<float> outputWeights,
        ArrayView<float> outputBias, int hiddenSize, ArrayView<float> logits)
    {
        this.hidden = hidden;
        this.outputWeights = outputWeights;
        this.outputBias = outputBias;
        this.hiddenSize = hiddenSize;
        this.logits = logits;
    }

    /// <summary>Writes one digit's unnormalized output for one image.</summary>
    /// <param name="index">The flattened batch row and digit index.</param>
    public void Execute(int index)
    {
        int row = index / 10;
        int digit = index % 10;
        float sum = outputBias[digit];
        for (int neuron = 0; neuron < hiddenSize; neuron++)
            sum += hidden[row * hiddenSize + neuron] * outputWeights[neuron * 10 + digit];
        logits[index] = sum;
    }
}

/// <summary>Computes stable softmax probabilities, cross-entropy gradients, and predictions.</summary>
public readonly struct SoftmaxKernel : IGpuForKernel
{
    /// <summary>The batch's row-major output logits.</summary>
    private readonly ArrayView<float> logits;
    /// <summary>The normalized class probabilities to write.</summary>
    private readonly ArrayView<float> probabilities;
    /// <summary>The original image labels, or an empty view for prediction only.</summary>
    private readonly ArrayView<int> labels;
    /// <summary>The selected image indices.</summary>
    private readonly ArrayView<int> order;
    /// <summary>The first selected image in the order buffer.</summary>
    private readonly int sampleOffset;
    /// <summary>The cross-entropy derivatives with respect to output logits.</summary>
    private readonly ArrayView<float> outputDeltas;
    /// <summary>The per-image metrics to reduce after the pass.</summary>
    private readonly ArrayView<NetworkMetric> metrics;
    /// <summary>The first batch row in the metrics buffer.</summary>
    private readonly int metricsOffset;
    /// <summary>The digit predictions to write.</summary>
    private readonly ArrayView<int> predictions;
    /// <summary>The first batch row in the predictions buffer.</summary>
    private readonly int predictionsOffset;

    /// <summary>Creates a probability, gradient, and metric pass for a batch.</summary>
    /// <param name="logits">The ten output logits per image.</param>
    /// <param name="probabilities">The ten probabilities per image to write.</param>
    /// <param name="labels">The original image labels, or an empty view for unlabeled images.</param>
    /// <param name="order">The selected image indices.</param>
    /// <param name="sampleOffset">The first batch image's position in the order buffer.</param>
    /// <param name="outputDeltas">The ten output derivatives per image to write.</param>
    /// <param name="metrics">The per-image loss and correctness values to write.</param>
    /// <param name="metricsOffset">The first batch row in the metrics buffer.</param>
    /// <param name="predictions">The digit predictions to write.</param>
    /// <param name="predictionsOffset">The first batch row in the predictions buffer.</param>
    public SoftmaxKernel(ArrayView<float> logits, ArrayView<float> probabilities,
        ArrayView<int> labels, ArrayView<int> order, int sampleOffset, ArrayView<float> outputDeltas,
        ArrayView<NetworkMetric> metrics, int metricsOffset, ArrayView<int> predictions, int predictionsOffset)
    {
        this.logits = logits;
        this.probabilities = probabilities;
        this.labels = labels;
        this.order = order;
        this.sampleOffset = sampleOffset;
        this.outputDeltas = outputDeltas;
        this.metrics = metrics;
        this.metricsOffset = metricsOffset;
        this.predictions = predictions;
        this.predictionsOffset = predictionsOffset;
    }

    /// <summary>Writes probabilities, derivatives, a prediction, and metrics for one image.</summary>
    /// <param name="index">The zero-based image index within the batch.</param>
    public void Execute(int index)
    {
        int first = index * 10;
        int predicted = 0;
        float maximum = logits[first];
        for (int digit = 1; digit < 10; digit++)
        {
            float value = logits[first + digit];
            if (value > maximum)
            {
                maximum = value;
                predicted = digit;
            }
        }

        // Subtracting the largest logit avoids overflowing exp during training.
        float sum = 0;
        for (int digit = 0; digit < 10; digit++)
        {
            float exponential = XMath.Exp(logits[first + digit] - maximum);
            probabilities[first + digit] = exponential;
            sum += exponential;
        }

        bool labeled = labels.Length > 0;
        int expected = labeled ? labels[order[sampleOffset + index]] : 0;
        for (int digit = 0; digit < 10; digit++)
        {
            float probability = probabilities[first + digit] / sum;
            probabilities[first + digit] = probability;
            outputDeltas[first + digit] = labeled ? probability - (digit == expected ? 1f : 0f) : 0;
        }

        predictions[predictionsOffset + index] = predicted;
        // Log-sum-exp computes cross entropy without taking log of a rounded-to-zero probability.
        metrics[metricsOffset + index] = labeled
            ? new NetworkMetric((maximum - logits[first + expected]) + XMath.Log(sum), predicted == expected ? 1 : 0)
            : default;
    }
}

/// <summary>Back-propagates output gradients through the original weights and ReLU activation.</summary>
public readonly struct HiddenGradientKernel : IGpuForKernel
{
    /// <summary>The batch's hidden activations used by the ReLU derivative.</summary>
    private readonly ArrayView<float> hidden;
    /// <summary>The batch's output logit derivatives.</summary>
    private readonly ArrayView<float> outputDeltas;
    /// <summary>The output weights before this batch's update.</summary>
    private readonly ArrayView<float> outputWeights;
    /// <summary>The hidden layer width.</summary>
    private readonly int hiddenSize;
    /// <summary>The hidden derivatives to write.</summary>
    private readonly ArrayView<float> hiddenDeltas;

    /// <summary>Creates the hidden layer's backward pass.</summary>
    /// <param name="hidden">The batch's hidden activations.</param>
    /// <param name="outputDeltas">The batch's output logit derivatives.</param>
    /// <param name="outputWeights">The output weights before their update.</param>
    /// <param name="hiddenSize">The hidden layer width.</param>
    /// <param name="hiddenDeltas">The hidden derivatives to write.</param>
    public HiddenGradientKernel(ArrayView<float> hidden, ArrayView<float> outputDeltas,
        ArrayView<float> outputWeights, int hiddenSize, ArrayView<float> hiddenDeltas)
    {
        this.hidden = hidden;
        this.outputDeltas = outputDeltas;
        this.outputWeights = outputWeights;
        this.hiddenSize = hiddenSize;
        this.hiddenDeltas = hiddenDeltas;
    }

    /// <summary>Writes the pre-activation derivative for one hidden neuron and image.</summary>
    /// <param name="index">The flattened batch row and hidden neuron index.</param>
    public void Execute(int index)
    {
        int row = index / hiddenSize;
        int neuron = index % hiddenSize;
        float sum = 0;
        if (hidden[index] > 0)
        {
            for (int digit = 0; digit < 10; digit++)
                sum += outputDeltas[row * 10 + digit] * outputWeights[neuron * 10 + digit];
        }
        hiddenDeltas[index] = sum;
    }
}

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
        float gradient = sum / batchCount + weightDecay * weight;
        float velocity = momentum * inputVelocity[index] - learningRate * gradient;
        inputVelocity[index] = velocity;
        inputWeights[index] = weight + velocity;
    }
}

/// <summary>Averages batch gradients and updates output weights using momentum and L2 weight decay.</summary>
public readonly struct UpdateOutputWeightsKernel : IGpuForKernel
{
    /// <summary>The batch's hidden activations.</summary>
    private readonly ArrayView<float> hidden;
    /// <summary>The batch's output logit derivatives.</summary>
    private readonly ArrayView<float> outputDeltas;
    /// <summary>The actual number of images in the batch.</summary>
    private readonly int batchCount;
    /// <summary>The hidden layer width.</summary>
    private readonly int hiddenSize;
    /// <summary>The output weights to update.</summary>
    private readonly ArrayView<float> outputWeights;
    /// <summary>The previous output weight update velocities.</summary>
    private readonly ArrayView<float> outputVelocity;
    /// <summary>The gradient step size.</summary>
    private readonly float learningRate;
    /// <summary>The fraction of each previous update retained.</summary>
    private readonly float momentum;
    /// <summary>The L2 coefficient applied to weights.</summary>
    private readonly float weightDecay;

    /// <summary>Creates an output weight update for a completed backward pass.</summary>
    /// <param name="hidden">The batch's hidden activations.</param>
    /// <param name="outputDeltas">The batch's output derivatives.</param>
    /// <param name="batchCount">The actual number of images in the batch.</param>
    /// <param name="hiddenSize">The hidden layer width.</param>
    /// <param name="outputWeights">The output weights to update.</param>
    /// <param name="outputVelocity">The output weight velocities to update.</param>
    /// <param name="learningRate">The gradient step size.</param>
    /// <param name="momentum">The fraction of the previous update retained.</param>
    /// <param name="weightDecay">The L2 regularization coefficient.</param>
    public UpdateOutputWeightsKernel(ArrayView<float> hidden, ArrayView<float> outputDeltas,
        int batchCount, int hiddenSize, ArrayView<float> outputWeights, ArrayView<float> outputVelocity,
        float learningRate, float momentum, float weightDecay)
    {
        this.hidden = hidden;
        this.outputDeltas = outputDeltas;
        this.batchCount = batchCount;
        this.hiddenSize = hiddenSize;
        this.outputWeights = outputWeights;
        this.outputVelocity = outputVelocity;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
    }

    /// <summary>Updates one output weight and its momentum velocity.</summary>
    /// <param name="index">The flattened hidden neuron and digit index.</param>
    public void Execute(int index)
    {
        int neuron = index / 10;
        int digit = index % 10;
        float sum = 0;
        for (int row = 0; row < batchCount; row++)
            sum += hidden[row * hiddenSize + neuron] * outputDeltas[row * 10 + digit];
        float weight = outputWeights[index];
        float gradient = sum / batchCount + weightDecay * weight;
        float velocity = momentum * outputVelocity[index] - learningRate * gradient;
        outputVelocity[index] = velocity;
        outputWeights[index] = weight + velocity;
    }
}

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

    /// <summary>Creates a bias update for either network layer.</summary>
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

/// <summary>Contains an image's cross-entropy loss and whether its prediction was correct.</summary>
public struct NetworkMetric
{
    /// <summary>The cross-entropy loss, summed when this value represents multiple images.</summary>
    public double Loss;
    /// <summary>The number of correctly classified images.</summary>
    public int Correct;

    /// <summary>Creates a device-compatible loss and correctness value.</summary>
    /// <param name="loss">The cross-entropy loss.</param>
    /// <param name="correct">The number of correct predictions.</param>
    public NetworkMetric(double loss, int correct)
    {
        Loss = loss;
        Correct = correct;
    }
}

/// <summary>Reduces image metrics to total cross-entropy loss and correct prediction count on the GPU.</summary>
public readonly struct NetworkMetricSum : IGpuReduction<NetworkMetric>
{
    /// <summary>Gets zero loss and zero correct predictions.</summary>
    public NetworkMetric Identity => default;

    /// <summary>Adds the loss and correctness values of two portions of a pass.</summary>
    /// <param name="left">The earlier portion's metrics.</param>
    /// <param name="right">The later portion's metrics.</param>
    /// <returns>The combined total loss and correct prediction count.</returns>
    public NetworkMetric Combine(NetworkMetric left, NetworkMetric right) =>
        new(left.Loss + right.Loss, left.Correct + right.Correct);
}
