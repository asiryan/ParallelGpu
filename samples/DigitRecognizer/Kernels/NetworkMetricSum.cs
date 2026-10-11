using ParallelGpu;

namespace DigitRecognizer;

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

