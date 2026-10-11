using ILGPU;
using ParallelGpu;

namespace DigitRecognizer;

/// <summary>Trains and runs a 784-input ReLU network with ten digit outputs using persistent GPU buffers.</summary>
public sealed class GpuNeuralNetwork : IDisposable
{
    /// <summary>The borrowed executor and the allocations owned by this network.</summary>
    private readonly Gpu gpu;
    private readonly List<IDisposable> allocations = [];
    private readonly int trainingCount;
    private readonly GpuBuffer<float> trainingPixels, inputWeights, inputBias, outputWeights, outputBias;
    private readonly GpuBuffer<float> inputVelocity, inputBiasVelocity, outputVelocity, outputBiasVelocity;
    private readonly GpuBuffer<float> hidden, logits, probabilities, outputDeltas, hiddenDeltas;
    private readonly GpuBuffer<int> trainingLabels, order, batchPredictions;
    private readonly GpuBuffer<NetworkMetric> batchMetrics;
    private bool disposed;

    /// <summary>Gets the number of hidden ReLU neurons.</summary>
    public int HiddenSize { get; }
    /// <summary>Gets the maximum number of images processed by one training batch.</summary>
    public int BatchSize { get; }
    /// <summary>Gets the fraction of the previous gradient update retained by momentum.</summary>
    public float Momentum { get; }
    /// <summary>Gets the L2 regularization coefficient applied to weights.</summary>
    public float WeightDecay { get; }

    /// <summary>Initializes weights and uploads the labeled dataset once for training and validation.</summary>
    /// <param name="gpu">The GPU executor; the caller retains ownership and disposes it.</param>
    /// <param name="trainingData">The normalized labeled images.</param>
    /// <param name="hiddenSize">The positive hidden layer width.</param>
    /// <param name="batchSize">The positive maximum number of images per batch.</param>
    /// <param name="seed">The seed used to initialize weights reproducibly.</param>
    /// <param name="momentum">The previous update fraction, from zero inclusive to one exclusive.</param>
    /// <param name="weightDecay">The finite nonnegative weight regularization coefficient.</param>
    /// <exception cref="ArgumentNullException">The executor or dataset is null.</exception>
    /// <exception cref="ArgumentException">The executor is not a GPU or the dataset has no labels.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A network setting is invalid.</exception>
    /// <exception cref="OverflowException">A configured buffer length exceeds Int32.MaxValue.</exception>
    public GpuNeuralNetwork(Gpu gpu, DigitDataset trainingData, int hiddenSize, int batchSize,
        int seed, float momentum = 0.9f, float weightDecay = 0.0001f)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        ArgumentNullException.ThrowIfNull(trainingData);
        if (!gpu.IsGpu)
            throw new ArgumentException("DigitRecognizer requires a GPU accelerator.", nameof(gpu));
        if (trainingData.Labels is null)
            throw new ArgumentException("Training images must have digit labels.", nameof(trainingData));
        foreach (int label in trainingData.Labels)
            if ((uint)label >= DigitDataset.ClassCount)
                throw new ArgumentException("Digit labels must be between 0 and 9.", nameof(trainingData));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        if (!float.IsFinite(momentum) || momentum < 0 || momentum >= 1)
            throw new ArgumentOutOfRangeException(nameof(momentum));
        if (!float.IsFinite(weightDecay) || weightDecay < 0)
            throw new ArgumentOutOfRangeException(nameof(weightDecay));

        this.gpu = gpu;
        trainingCount = trainingData.Count;
        HiddenSize = hiddenSize;
        BatchSize = batchSize;
        Momentum = momentum;
        WeightDecay = weightDecay;
        int inputWeightCount = checked(DigitDataset.InputSize * hiddenSize);
        int outputWeightCount = checked(hiddenSize * DigitDataset.ClassCount);
        int hiddenCount = checked(batchSize * hiddenSize);
        int outputCount = checked(batchSize * DigitDataset.ClassCount);
        var random = new Random(seed);

        try
        {
            trainingPixels = Own(gpu.Upload(trainingData.Pixels));
            trainingLabels = Own(gpu.Upload(trainingData.Labels));
            order = Own(gpu.Allocate<int>(trainingCount));
            // He initialization for ReLU, and Xavier initialization for the linear output layer.
            inputWeights = Own(gpu.Upload(RandomWeights(inputWeightCount, Math.Sqrt(6.0 / DigitDataset.InputSize), random)));
            outputWeights = Own(gpu.Upload(RandomWeights(outputWeightCount, Math.Sqrt(6.0 / (hiddenSize + DigitDataset.ClassCount)), random)));
            inputBias = Own(gpu.Allocate<float>(hiddenSize));
            outputBias = Own(gpu.Allocate<float>(DigitDataset.ClassCount));
            inputVelocity = Own(gpu.Allocate<float>(inputWeightCount));
            outputVelocity = Own(gpu.Allocate<float>(outputWeightCount));
            inputBiasVelocity = Own(gpu.Allocate<float>(hiddenSize));
            outputBiasVelocity = Own(gpu.Allocate<float>(DigitDataset.ClassCount));
            hidden = Own(gpu.Allocate<float>(hiddenCount));
            hiddenDeltas = Own(gpu.Allocate<float>(hiddenCount));
            logits = Own(gpu.Allocate<float>(outputCount));
            probabilities = Own(gpu.Allocate<float>(outputCount));
            outputDeltas = Own(gpu.Allocate<float>(outputCount));
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
    public void WarmUp()
    {
        ThrowIfDisposed();
        GpuBuffer<float>[] parameters = [inputWeights, inputBias, outputWeights, outputBias,
            inputVelocity, inputBiasVelocity, outputVelocity, outputBiasVelocity];
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
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x50474E31); // PGN1: sample model format and version.
        writer.Write(DigitDataset.InputSize);
        writer.Write(HiddenSize);
        writer.Write(DigitDataset.ClassCount);
        foreach (var buffer in new[] { inputWeights, inputBias, outputWeights, outputBias })
            foreach (float value in buffer.ToArray())
                writer.Write(value);
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
    private void TrainBatch(int offset, int count, float learningRate)
    {
        Forward(trainingPixels.View, trainingLabels.View, order.View, offset, count,
            batchMetrics.View, 0, batchPredictions.View, 0);
        // Compute all hidden derivatives before changing the output weights they depend on.
        gpu.For(0, count * HiddenSize, new HiddenGradientKernel(hidden.View, outputDeltas.View,
            outputWeights.View, HiddenSize, hiddenDeltas.View));
        gpu.For(0, inputWeights.Length, new UpdateInputWeightsKernel(trainingPixels.View, order.View,
            offset, count, HiddenSize, hiddenDeltas.View, inputWeights.View, inputVelocity.View,
            learningRate, Momentum, WeightDecay));
        gpu.For(0, outputWeights.Length, new UpdateOutputWeightsKernel(hidden.View, outputDeltas.View,
            count, HiddenSize, outputWeights.View, outputVelocity.View, learningRate, Momentum, WeightDecay));
        gpu.For(0, HiddenSize, new UpdateBiasKernel(hiddenDeltas.View, HiddenSize, count,
            inputBias.View, inputBiasVelocity.View, learningRate, Momentum));
        gpu.For(0, DigitDataset.ClassCount, new UpdateBiasKernel(outputDeltas.View, DigitDataset.ClassCount,
            count, outputBias.View, outputBiasVelocity.View, learningRate, Momentum));
    }

    /// <summary>Runs the two dense layers and stable softmax on one batch.</summary>
    private void Forward(ArrayView<float> pixels, ArrayView<int> labels, ArrayView<int> indices,
        int offset, int count, ArrayView<NetworkMetric> metrics, int metricsOffset,
        ArrayView<int> predictions, int predictionsOffset)
    {
        gpu.For(0, count * HiddenSize, new ForwardHiddenKernel(pixels, indices, offset, HiddenSize,
            inputWeights.View, inputBias.View, hidden.View));
        gpu.For(0, count * DigitDataset.ClassCount, new ForwardOutputKernel(hidden.View,
            outputWeights.View, outputBias.View, HiddenSize, logits.View));
        gpu.For(0, count, new SoftmaxKernel(logits.View, probabilities.View, labels, indices, offset,
            outputDeltas.View, metrics, metricsOffset, predictions, predictionsOffset));
    }

    /// <summary>Validates selected row indices and copies their order to the persistent device buffer.</summary>
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
    private GpuBuffer<T> Own<T>(GpuBuffer<T> buffer) where T : unmanaged
    {
        allocations.Add(buffer);
        return buffer;
    }

    /// <summary>Creates reproducible uniform weights within the symmetric initialization interval.</summary>
    private static float[] RandomWeights(int count, double limit, Random random)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)((random.NextDouble() * 2 - 1) * limit);
        return values;
    }

    /// <summary>Rejects calls after the owned buffers have been released.</summary>
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
}

/// <summary>Contains classification measurements for a labeled image selection.</summary>
/// <param name="Count">The number of evaluated images.</param>
/// <param name="MeanLoss">The mean cross-entropy loss.</param>
/// <param name="Accuracy">The fraction of correctly classified images, from zero through one.</param>
public readonly record struct EvaluationResult(int Count, double MeanLoss, double Accuracy);
