namespace MatrixMultiplication;

public static class Matrices
{
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

    public static float[][] Random(int rows, int columns, int seed)
    {
        float[][] matrix = Create(rows, columns);
        var random = new Random(seed);
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                matrix[row][column] = random.NextSingle() * 2 - 1;
        return matrix;
    }

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

    internal static (int Rows, int Inner, int Columns) ProductShape(float[][] a, float[][] b)
    {
        var left = Shape(a, nameof(a));
        var right = Shape(b, nameof(b));
        if (left.Columns != right.Rows)
            throw new ArgumentException("A's column count must equal B's row count.", nameof(b));
        _ = checked(left.Rows * right.Columns);
        return (left.Rows, left.Columns, right.Columns);
    }

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
