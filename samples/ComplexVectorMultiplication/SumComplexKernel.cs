using System.Numerics;
using ILGPU;
using ParallelGpu;

namespace ComplexVectorMultiplication;

/// <summary>Reduces each batch of adjacent device values to one partial complex sum.</summary>
public readonly struct SumComplexKernel : IGpuForKernel
{
    /// <summary>The device values consumed during this reduction pass.</summary>
    private readonly ArrayView<Complex> input;
    /// <summary>The separate device buffer receiving one result per batch.</summary>
    private readonly ArrayView<Complex> output;

    /// <summary>Creates a reduction pass with separate input and output allocations.</summary>
    /// <param name="input">The products or partial sums to reduce.</param>
    /// <param name="output">The destination for the reduced batches.</param>
    public SumComplexKernel(ArrayView<Complex> input, ArrayView<Complex> output)
    {
        this.input = input;
        this.output = output;
    }

    /// <summary>Sums one batch, including a shorter batch at the end of the input.</summary>
    /// <param name="index">The index of the batch and its output element.</param>
    public void Execute(int index)
    {
        int start = index * VectorProduct.BatchSize;
        int count = Math.Min(VectorProduct.BatchSize, (int)input.Length - start);
        Complex sum = Complex.Zero;
        for (int i = 0; i < count; i++)
            sum += input[start + i];
        output[index] = sum;
    }
}
