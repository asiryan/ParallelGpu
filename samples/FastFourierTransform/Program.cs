using System.Diagnostics;
using System.Numerics;
using FastFourierTransform;
using ParallelGpu;

// Matrix of complex doubles.
const int Rows = 4096;
const int Columns = 4096;
const bool Normalized = true;

try
{
    using var gpu = new Gpu(new GpuOptions
    {
        Backend = GpuBackend.Auto,
        AllowCpuFallback = false
    });

    var input = new Complex[Rows, Columns];
    var random = new Random(12345);
    for (int row = 0; row < Rows; row++)
        for (int column = 0; column < Columns; column++)
            input[row, column] = new Complex(random.NextDouble() - 0.5, random.NextDouble() - 0.5);

    Console.WriteLine($"Device: {gpu.DeviceName} ({gpu.Backend})");
    Console.WriteLine($"Input: {Rows} x {Columns} complex values");
    Console.WriteLine("Operation: forward 2D FFT, exp(-2πi * (ky*y/rows + kx*x/columns))");
    Console.WriteLine("Algorithm: radix-2 Cooley–Tukey FFTs over rows and columns");
    Console.WriteLine($"Normalization: {(Normalized ? "1 / sqrt(rows * columns)" : "none")}");

    // Compile the reusable radix-2 kernels before measuring the full transform.
    FourierTransform.Forward2D(gpu, new Complex[8, 8], Normalized);

    var timer = Stopwatch.StartNew();
    Complex[,] spectrum = FourierTransform.Forward2D(gpu, input, Normalized);
    timer.Stop();

    Console.WriteLine($"{gpu.Backend} FFT, including allocations and array transfers: {timer.Elapsed.TotalMilliseconds:F3} ms");
    Console.WriteLine($"Spectrum[0, 0]: {spectrum[0, 0]}");
    Console.WriteLine($"Spectrum[1, 1]: {spectrum[1, 1]}");
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidOperationException or OverflowException)
{
    Console.Error.WriteLine($"FastFourierTransform failed: {error.Message}");
    return 1;
}
