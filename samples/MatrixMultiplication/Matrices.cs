namespace MatrixMultiplication;

/// <summary>Creates matrices and validates their dimensions and multiplication results.</summary>
public static class Matrices
{
    /// <summary>Creates a zero-filled matrix with independent rows of equal length.</summary>
    /// <param name="rows">The positive number of rows.</param>
    /// <param name="columns">The positive number of columns.</param>
    /// <returns>A rectangular matrix represented by a jagged array.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Either dimension is zero or negative.</exception>
    /// <exception cref="OverflowException">The element count exceeds Int32.MaxValue.</exception>
    public static float[][] Create(int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        _ = checked(rows * columns);

        var matrix = new float[rows][];
        for (int row = 0; row < rows; row++)
            matrix[row] = new float[columns];
        return matrix;
    }

    /// <summary>Creates a reproducible matrix with random values in [-1, 1).</summary>
    /// <param name="rows">The positive number of rows.</param>
    /// <param name="columns">The positive number of columns.</param>
    /// <param name="seed">The random generator seed.</param>
    /// <returns>A matrix containing the generated values.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Either dimension is zero or negative.</exception>
    /// <exception cref="OverflowException">The element count exceeds Int32.MaxValue.</exception>
    public static float[][] Random(int rows, int columns, int seed)
    {
        float[][] matrix = Create(rows, columns);
        var random = new Random(seed);
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                matrix[row][column] = random.NextSingle() * 2 - 1;
        return matrix;
    }

    /// <summary>Validates each matrix element with an absolute and relative tolerance.</summary>
    /// <param name="expected">The CPU reference matrix.</param>
    /// <param name="actual">The GPU result matrix.</param>
    /// <returns>The maximum absolute element error.</returns>
    /// <exception cref="ArgumentNullException">Either matrix is null.</exception>
    /// <exception cref="ArgumentException">A matrix is empty or ragged, or the dimensions differ.</exception>
    /// <exception cref="InvalidOperationException">An element is nonfinite or differs beyond tolerance.</exception>
    public static double CheckEqual(float[][] expected, float[][] actual)
    {
        var shape = Shape(expected, nameof(expected));
        if (shape != Shape(actual, nameof(actual)))
            throw new ArgumentException("Matrix dimensions do not match.", nameof(actual));

        double maxError = 0;
        for (int row = 0; row < shape.Rows; row++)
        {
            for (int column = 0; column < shape.Columns; column++)
            {
                double reference = expected[row][column];
                double value = actual[row][column];
                double error = Math.Abs(value - reference);
                double tolerance = 1e-4 + 1e-4 * Math.Abs(reference);
                if (!double.IsFinite(reference) || !double.IsFinite(value) || error > tolerance)
                    throw new InvalidOperationException($"Results differ at [{row}, {column}]: {reference} vs {value}.");
                maxError = Math.Max(maxError, error);
            }
        }
        return maxError;
    }

    /// <summary>Validates matrix compatibility and determines the multiplication dimensions.</summary>
    /// <param name="a">The left matrix.</param>
    /// <param name="b">The right matrix.</param>
    /// <returns>The output row count, shared inner dimension, and output column count.</returns>
    /// <exception cref="ArgumentNullException">Either matrix is null.</exception>
    /// <exception cref="ArgumentException">A matrix is empty or ragged, or the inner dimensions differ.</exception>
    /// <exception cref="OverflowException">The output element count exceeds Int32.MaxValue.</exception>
    internal static (int Rows, int Inner, int Columns) ProductShape(float[][] a, float[][] b)
    {
        var left = Shape(a, nameof(a));
        var right = Shape(b, nameof(b));
        if (left.Columns != right.Rows)
            throw new ArgumentException("A's column count must equal B's row count.", nameof(b));
        _ = checked(left.Rows * right.Columns);
        return (left.Rows, left.Columns, right.Columns);
    }

    /// <summary>Checks that a jagged array represents a nonempty rectangular matrix.</summary>
    /// <param name="matrix">The matrix to validate.</param>
    /// <param name="name">The parameter name included in validation exceptions.</param>
    /// <returns>The common row and column counts.</returns>
    /// <exception cref="ArgumentNullException">The matrix is null.</exception>
    /// <exception cref="ArgumentException">The matrix is empty, contains a null row, or has inconsistent row lengths.</exception>
    private static (int Rows, int Columns) Shape(float[][] matrix, string name)
    {
        ArgumentNullException.ThrowIfNull(matrix, name);
        if (matrix.Length == 0 || matrix[0] is null || matrix[0].Length == 0)
            throw new ArgumentException("A matrix must have at least one row and column.", name);
        int columns = matrix[0].Length;
        foreach (float[] row in matrix)
            if (row is null || row.Length != columns)
                throw new ArgumentException("Every matrix row must have the same nonzero length.", name);
        return (matrix.Length, columns);
    }
}
