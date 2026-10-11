using ILGPU;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Trains and runs a configurable multilayer ReLU network for digits using persistent GPU buffers.</summary>
public sealed class GpuNeuralNetwork : IDisposable
{
    /// <summary>The borrowed executor used for buffer allocation and every network kernel.</summary>
    private readonly Gpu gpu;

    /// <summary>The owned buffers, recorded in allocation order for complete failure cleanup and disposal.</summary>
    private readonly List<IDisposable> allocations = [];

    /// <summary>The number of labeled images available for training and held-out evaluation.</summary>
    private readonly int trainingCount;

    /// <summary>The normalized training images uploaded once and retained on the GPU.</summary>
    private readonly GpuBuffer<float> trainingPixels;

    /// <summary>The current batch's ten softmax probabilities per image.</summary>
    private readonly GpuBuffer<float> probabilities;

    /// <summary>The dense layers in forward-pass order, including the final linear output layer.</summary>
    private readonly GpuNetworkLayer[] layers;

    /// <summary>The digit labels corresponding to the uploaded training images.</summary>
    private readonly GpuBuffer<int> trainingLabels;

    /// <summary>The selected image indices, refreshed for each training or evaluation pass.</summary>
    private readonly GpuBuffer<int> order;

    /// <summary>The current batch's predictions when a full prediction buffer is unnecessary.</summary>
    private readonly GpuBuffer<int> batchPredictions;

    /// <summary>The batch metric workspace used by training, prediction, and kernel warm-up.</summary>
    private readonly GpuBuffer<NetworkMetric> batchMetrics;

    /// <summary>Whether the network's owned allocations have been released.</summary>
    private bool disposed;

    /// <summary>Gets the widths of the hidden ReLU layers in forward-pass order.</summary>
    public IReadOnlyList<int> HiddenLayerSizes { get; }
    /// <summary>Gets the maximum number of images processed by one training batch.</summary>
    public int BatchSize { get; }
    /// <summary>Gets the fraction of the previous gradient update retained by momentum.</summary>
    public float Momentum { get; }
    /// <summary>Gets the L2 regularization coefficient applied to weights.</summary>
    public float WeightDecay { get; }

    /// <summary>Initializes weights and uploads the labeled dataset once for training and validation.</summary>
    /// <param name="gpu">The GPU executor; the caller retains ownership and disposes it.</param>
    /// <param name="trainingData">The normalized labeled images.</param>
    /// <param name="hiddenLayerSizes">The positive widths of one or more hidden ReLU layers.</param>
    /// <param name="batchSize">The positive maximum number of images per batch.</param>
    /// <param name="seed">The seed used to initialize weights reproducibly.</param>
    /// <param name="momentum">The previous update fraction, from zero inclusive to one exclusive.</param>
    /// <param name="weightDecay">The finite nonnegative weight regularization coefficient.</param>
    /// <exception cref="ArgumentNullException">The executor, dataset, or hidden layer array is null.</exception>
    /// <exception cref="ArgumentException">The executor is not a GPU, labels are missing or invalid, or no hidden layers are specified.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A network setting is invalid.</exception>
    /// <exception cref="OverflowException">A configured buffer length exceeds Int32.MaxValue.</exception>
    public GpuNeuralNetwork(Gpu gpu, DigitDataset trainingData, int[] hiddenLayerSizes, int batchSize,
        int seed, float momentum = 0.9f, float weightDecay = 0.0001f)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        ArgumentNullException.ThrowIfNull(trainingData);
        ArgumentNullException.ThrowIfNull(hiddenLayerSizes);
        if (!gpu.IsGpu)
            throw new ArgumentException("DigitRecognizer requires a GPU accelerator.", nameof(gpu));
        if (trainingData.Labels is null)
            throw new ArgumentException("Training images must have digit labels.", nameof(trainingData));
        foreach (int label in trainingData.Labels)
            if ((uint)label >= DigitDataset.ClassCount)
                throw new ArgumentException("Digit labels must be between 0 and 9.", nameof(trainingData));
        if (hiddenLayerSizes.Length == 0)
            throw new ArgumentException("Specify at least one hidden layer.", nameof(hiddenLayerSizes));
        foreach (int width in hiddenLayerSizes)
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(hiddenLayerSizes), "Hidden layer widths must be positive.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        if (!float.IsFinite(momentum) || momentum < 0 || momentum >= 1)
            throw new ArgumentOutOfRangeException(nameof(momentum));
        if (!float.IsFinite(weightDecay) || weightDecay < 0)
            throw new ArgumentOutOfRangeException(nameof(weightDecay));

        this.gpu = gpu;
        trainingCount = trainingData.Count;
        HiddenLayerSizes = Array.AsReadOnly(hiddenLayerSizes.ToArray());
        BatchSize = batchSize;
        Momentum = momentum;
        WeightDecay = weightDecay;
        layers = new GpuNetworkLayer[checked(hiddenLayerSizes.Length + 1)];
        int outputCount = checked(batchSize * DigitDataset.ClassCount);
        var random = new Random(seed);

        try
        {
            trainingPixels = Own(gpu.Upload(trainingData.Pixels));
            trainingLabels = Own(gpu.Upload(trainingData.Labels));
            order = Own(gpu.Allocate<int>(trainingCount));
            int inputSize = DigitDataset.InputSize;
            for (int index = 0; index < layers.Length; index++)
            {
                bool outputLayer = index == layers.Length - 1;
                int width = outputLayer ? DigitDataset.ClassCount : HiddenLayerSizes[index];
                int weightCount = checked(inputSize * width);
                int activationCount = checked(batchSize * width);
                // He initialization for each ReLU layer, and Xavier for the linear output layer.
                double limit = outputLayer
                    ? Math.Sqrt(6.0 / ((double)inputSize + width))
                    : Math.Sqrt(6.0 / inputSize);
                layers[index] = new GpuNetworkLayer(inputSize, width,
                    Own(gpu.Upload(RandomWeights(weightCount, limit, random))),
                    Own(gpu.Allocate<float>(width)),
                    Own(gpu.Allocate<float>(weightCount)),
                    Own(gpu.Allocate<float>(width)),
                    Own(gpu.Allocate<float>(activationCount)),
                    Own(gpu.Allocate<float>(activationCount)));
                inputSize = width;
            }
            probabilities = Own(gpu.Allocate<float>(outputCount));
            batchMetrics = Own(gpu.Allocate<NetworkMetric>(batchSize));
            batchPredictions = Own(gpu.Allocate<int>(batchSize));
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Compiles every training kernel and metric reduction, then restores all parameters and momentum.</summary>
    /// <remarks>A temporary training step is discarded so warm-up cannot affect model quality.</remarks>
    /// <exception cref="ObjectDisposedException">The network has been disposed.</exception>
    public void WarmUp()
    {
        ThrowIfDisposed();
        GpuBuffer<float>[] parameters = layers.SelectMany(layer => new[]
            { layer.Weights, layer.Bias, layer.WeightVelocity, layer.BiasVelocity }).ToArray();
        float[][] snapshots = parameters.Select(buffer => buffer.ToArray()).ToArray();
        int count = Math.Min(BatchSize, trainingCount);
        order.CopyFrom(0, Enumerable.Range(0, count).ToArray());
        try
        {
            TrainBatch(0, count, 0.01f);
            gpu.Reduce(batchMetrics, new NetworkMetricSum());
        }
        finally
        {
            for (int i = 0; i < parameters.Length; i++)
                parameters[i].CopyFrom(snapshots[i]);
        }
    }

    /// <summary>Trains once over the selected images in their supplied order.</summary>
    /// <param name="sampleIndices">Only training row indices, shuffled by the caller before each epoch.</param>
    /// <param name="learningRate">The finite positive gradient step size.</param>
    /// <exception cref="ArgumentNullException">The selected row array is null.</exception>
    /// <exception cref="ArgumentException">The selected rows are empty or outside the labeled dataset.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The learning rate is not finite and positive.</exception>
    /// <exception cref="ObjectDisposedException">The network has been disposed.</exception>
    public void TrainEpoch(int[] sampleIndices, float learningRate)
    {
        PrepareOrder(sampleIndices);
        if (!float.IsFinite(learningRate) || learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate));
        for (int offset = 0; offset < sampleIndices.Length;)
        {
            int count = Math.Min(BatchSize, sampleIndices.Length - offset);
            TrainBatch(offset, count, learningRate);
            offset += count;
        }
    }

    /// <summary>Measures loss and accuracy on selected labeled images without updating parameters.</summary>
    /// <param name="sampleIndices">The held-out row indices to evaluate.</param>
    /// <returns>The image count, mean cross-entropy loss, and classification accuracy.</returns>
    /// <exception cref="ArgumentNullException">The selected row array is null.</exception>
    /// <exception cref="ArgumentException">The selected rows are empty or outside the labeled dataset.</exception>
    /// <exception cref="InvalidOperationException">The network produced nonfinite loss.</exception>
    /// <exception cref="ObjectDisposedException">The network has been disposed.</exception>
    public EvaluationResult Evaluate(int[] sampleIndices)
    {
        PrepareOrder(sampleIndices);
        // Allocate exactly one metric per selected image, so a tail batch cannot leave stale values in the reduction.
        using var metrics = gpu.Allocate<NetworkMetric>(sampleIndices.Length);
        for (int offset = 0; offset < sampleIndices.Length;)
        {
            int count = Math.Min(BatchSize, sampleIndices.Length - offset);
            Forward(trainingPixels.View, trainingLabels.View, order.View, offset, count,
                metrics.View, offset, batchPredictions.View, 0);
            offset += count;
        }
        NetworkMetric total = gpu.Reduce(metrics, new NetworkMetricSum());
        double loss = total.Loss / sampleIndices.Length;
        if (!double.IsFinite(loss))
            throw new InvalidOperationException("Network loss became nonfinite; reduce the learning rate.");
        return new(sampleIndices.Length, loss, (double)total.Correct / sampleIndices.Length);
    }

    /// <summary>Predicts selected rows of the already uploaded labeled dataset.</summary>
    /// <param name="sampleIndices">The image row indices to classify.</param>
    /// <returns>One predicted digit for each selected row, in the supplied order.</returns>
    /// <exception cref="ArgumentNullException">The selected row array is null.</exception>
    /// <exception cref="ArgumentException">The selected rows are empty or outside the labeled dataset.</exception>
    /// <exception cref="ObjectDisposedException">The network has been disposed.</exception>
    public int[] PredictTrainingRows(int[] sampleIndices)
    {
        PrepareOrder(sampleIndices);
        return PredictViews(trainingPixels.View, trainingLabels.View, order.View, sampleIndices.Length);
    }

    /// <summary>Uploads another dataset and predicts all of its images without training.</summary>
    /// <param name="data">The normalized images to classify; labels are not required.</param>
    /// <returns>One predicted digit per image in original CSV order.</returns>
    /// <exception cref="ArgumentNullException">The dataset is null.</exception>
    /// <exception cref="ObjectDisposedException">The network has been disposed.</exception>
    public int[] Predict(DigitDataset data)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(data);
        using var pixels = gpu.Upload(data.Pixels);
        // Prediction does not use labels, even if the caller supplied a labeled dataset.
        using var labels = gpu.Allocate<int>(0);
        using var indices = gpu.Upload(Enumerable.Range(0, data.Count).ToArray());
        return PredictViews(pixels.View, labels.View, indices.View, data.Count);
    }

    /// <summary>Saves the trained layer dimensions, weights, and biases as a binary model.</summary>
    /// <param name="path">The destination filename; its parent directory must exist.</param>
    /// <exception cref="IOException">The file cannot be written.</exception>
    /// <exception cref="ObjectDisposedException">The network has been disposed.</exception>
    public void Save(string path)
    {
        ThrowIfDisposed();
        NetworkModelWriter.Write(path, layers);
    }

    /// <summary>Releases all dataset, parameter, and workspace buffers owned by this network.</summary>
    /// <remarks>The borrowed GPU executor remains owned by the caller.</remarks>
    public void Dispose()
    {
        if (disposed)
            return;
        for (int i = allocations.Count - 1; i >= 0; i--)
            allocations[i].Dispose();
        allocations.Clear();
        disposed = true;
    }

    /// <summary>Runs inference over ordered rows and downloads only the resulting digit predictions.</summary>
    /// <param name="pixels">The normalized images already resident on the device.</param>
    /// <param name="labels">Their labels, or an empty view for unlabeled prediction.</param>
    /// <param name="indices">The selected image indices in prediction order.</param>
    /// <param name="count">The positive number of selected images.</param>
    /// <returns>One predicted digit for each selected image, preserving the supplied order.</returns>
    private int[] PredictViews(ArrayView<float> pixels, ArrayView<int> labels, ArrayView<int> indices, int count)
    {
        using var predictions = gpu.Allocate<int>(count);
        for (int offset = 0; offset < count;)
        {
            int batchCount = Math.Min(BatchSize, count - offset);
            Forward(pixels, labels, indices, offset, batchCount,
                batchMetrics.View, 0, predictions.View, offset);
            offset += batchCount;
        }
        return predictions.ToArray();
    }

    /// <summary>Executes one mini-batch's forward pass, backward pass, and momentum updates.</summary>
    /// <param name="offset">The first selected row's position in the persistent order buffer.</param>
    /// <param name="count">The actual batch size, including a final incomplete batch.</param>
    /// <param name="learningRate">The validated positive gradient step size.</param>
    private void TrainBatch(int offset, int count, float learningRate)
    {
        Forward(trainingPixels.View, trainingLabels.View, order.View, offset, count,
            batchMetrics.View, 0, batchPredictions.View, 0);
        // Back-propagate through every hidden layer before changing any weights used by its derivatives.
        for (int index = layers.Length - 2; index >= 0; index--)
        {
            GpuNetworkLayer layer = layers[index];
            GpuNetworkLayer next = layers[index + 1];
            gpu.For(0, count * layer.OutputSize, new BackwardHiddenKernel(layer.Activations.View,
                next.Deltas.View, next.Weights.View, layer.OutputSize, next.OutputSize, layer.Deltas.View));
        }
        for (int index = 0; index < layers.Length; index++)
        {
            GpuNetworkLayer layer = layers[index];
            if (index == 0)
                gpu.For(0, layer.Weights.Length, new UpdateInputWeightsKernel(trainingPixels.View, order.View,
                    offset, count, layer.OutputSize, layer.Deltas.View, layer.Weights.View, layer.WeightVelocity.View,
                    learningRate, Momentum, WeightDecay));
            else
                gpu.For(0, layer.Weights.Length, new UpdateDenseWeightsKernel(layers[index - 1].Activations.View,
                    layer.Deltas.View, count, layer.InputSize, layer.OutputSize, layer.Weights.View,
                    layer.WeightVelocity.View, learningRate, Momentum, WeightDecay));
            gpu.For(0, layer.OutputSize, new UpdateBiasKernel(layer.Deltas.View, layer.OutputSize, count,
                layer.Bias.View, layer.BiasVelocity.View, learningRate, Momentum));
        }
    }

    /// <summary>Runs all dense layers and stable softmax on one batch.</summary>
    /// <param name="pixels">The normalized images already resident on the device.</param>
    /// <param name="labels">The original image labels, or an empty view for prediction only.</param>
    /// <param name="indices">The selected image indices for this pass.</param>
    /// <param name="offset">The first batch row's position in the selected index view.</param>
    /// <param name="count">The actual number of images in this batch.</param>
    /// <param name="metrics">The per-image metric buffer to write.</param>
    /// <param name="metricsOffset">The first batch row's destination in the metric buffer.</param>
    /// <param name="predictions">The predicted digit buffer to write.</param>
    /// <param name="predictionsOffset">The first batch row's destination in the prediction buffer.</param>
    private void Forward(ArrayView<float> pixels, ArrayView<int> labels, ArrayView<int> indices,
        int offset, int count, ArrayView<NetworkMetric> metrics, int metricsOffset,
        ArrayView<int> predictions, int predictionsOffset)
    {
        GpuNetworkLayer first = layers[0];
        gpu.For(0, count * first.OutputSize, new ForwardHiddenKernel(pixels, indices, offset, first.OutputSize,
            first.Weights.View, first.Bias.View, first.Activations.View));
        for (int index = 1; index < layers.Length; index++)
        {
            GpuNetworkLayer layer = layers[index];
            gpu.For(0, count * layer.OutputSize, new ForwardDenseKernel(layers[index - 1].Activations.View,
                layer.Weights.View, layer.Bias.View, layer.InputSize, layer.OutputSize,
                index < layers.Length - 1, layer.Activations.View));
        }
        GpuNetworkLayer output = layers[^1];
        gpu.For(0, count, new SoftmaxKernel(output.Activations.View, probabilities.View, labels, indices, offset,
            output.Deltas.View, metrics, metricsOffset, predictions, predictionsOffset));
    }

    /// <summary>Validates selected row indices and copies their order to the persistent device buffer.</summary>
    /// <param name="sampleIndices">The image row indices in the caller's requested order.</param>
    /// <exception cref="ArgumentNullException">The selected row array is null.</exception>
    /// <exception cref="ArgumentException">The selection is empty, too large, or contains an invalid row index.</exception>
    /// <exception cref="ObjectDisposedException">The network has been disposed.</exception>
    private void PrepareOrder(int[] sampleIndices)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(sampleIndices);
        if (sampleIndices.Length == 0 || sampleIndices.Length > trainingCount)
            throw new ArgumentException("Select between one and the full labeled image count.", nameof(sampleIndices));
        foreach (int index in sampleIndices)
            if ((uint)index >= (uint)trainingCount)
                throw new ArgumentException("An image index is outside the labeled dataset.", nameof(sampleIndices));
        order.CopyFrom(0, sampleIndices);
    }

    /// <summary>Registers one allocation for disposal, including partial constructor failure.</summary>
    /// <typeparam name="T">The unmanaged device element type.</typeparam>
    /// <param name="buffer">A newly allocated buffer whose ownership transfers to this network.</param>
    /// <returns>The registered buffer for assignment to a layer or workspace.</returns>
    private GpuBuffer<T> Own<T>(GpuBuffer<T> buffer) where T : unmanaged
    {
        allocations.Add(buffer);
        return buffer;
    }

    /// <summary>Creates reproducible uniform weights within the symmetric initialization interval.</summary>
    /// <param name="count">The positive number of weights to initialize.</param>
    /// <param name="limit">The positive upper magnitude of the initialization interval.</param>
    /// <param name="random">The shared seeded generator advanced in layer order.</param>
    /// <returns>The weights sampled uniformly between the negative and positive limits.</returns>
    private static float[] RandomWeights(int count, double limit, Random random)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)((random.NextDouble() * 2 - 1) * limit);
        return values;
    }

    /// <summary>Rejects calls after the owned buffers have been released.</summary>
    /// <exception cref="ObjectDisposedException">The network has been disposed.</exception>
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
}
