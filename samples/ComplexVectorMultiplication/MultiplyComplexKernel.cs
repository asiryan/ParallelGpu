using System.Numerics;
using ILGPU;
using ParallelGpu;

namespace ComplexVectorMultiplication;

/// <summary>Multiplies each left-vector element by the corresponding right-vector element.</summary>
public readonly struct MultiplyComplexKernel : IGpuForEachKernel<Complex>
{
    /// <summary>The device view of the unchanged right vector.</summary>
    private readonly ArrayView<Complex> right;

    /// <summary>Creates a kernel using the uploaded right vector.</summary>
    /// <param name="right">The device view containing one multiplier per element.</param>
    public MultiplyComplexKernel(ArrayView<Complex> right) => this.right = right;

    /// <summary>Stores one complex product in place in the uploaded left-vector buffer.</summary>
    /// <param name="index">The element index shared by both vectors.</param>
    /// <param name="value">The left value replaced by its product.</param>
    public void Execute(int index, ref Complex value) => value *= right[index];
}
