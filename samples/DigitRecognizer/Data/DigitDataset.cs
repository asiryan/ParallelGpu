namespace DigitRecognizer;

/// <summary>Stores normalized, row-major images and their optional digit labels on the host.</summary>
public sealed class DigitDataset
{
    /// <summary>The width and height of every grayscale digit image.</summary>
    public const int ImageWidth = 28;

    /// <summary>The number of pixels supplied to the network for one image.</summary>
    public const int InputSize = ImageWidth * ImageWidth;

    /// <summary>The number of possible digit labels, from zero through nine.</summary>
    public const int ClassCount = 10;

    /// <summary>Gets the flat image array, with one consecutive block of normalized pixels per image.</summary>
    public float[] Pixels { get; }

    /// <summary>Gets the digit labels, or null for the unlabeled Kaggle test set.</summary>
    public int[]? Labels { get; }

    /// <summary>Gets the number of complete images in this dataset.</summary>
    public int Count { get; }

    /// <summary>Validates and wraps complete image arrays without copying their contents.</summary>
    /// <param name="pixels">Row-major pixels normalized to the range from zero through one.</param>
    /// <param name="labels">One label per image, or null when the dataset is unlabeled.</param>
    /// <exception cref="ArgumentNullException">The pixel array is null.</exception>
    /// <remarks>The caller must keep the wrapped arrays within the validated pixel and label ranges.</remarks>
    /// <exception cref="ArgumentException">Image dimensions or label counts do not match, a pixel is not finite and within zero through one, or a label is outside zero through nine.</exception>
    public DigitDataset(float[] pixels, int[]? labels)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        if (pixels.Length == 0 || pixels.Length % InputSize != 0)
            throw new ArgumentException("Pixels must contain at least one complete 28 x 28 image.", nameof(pixels));
        if (labels is not null && labels.Length != pixels.Length / InputSize)
            throw new ArgumentException("The number of labels must match the number of images.", nameof(labels));
        foreach (float pixel in pixels)
            if (!float.IsFinite(pixel) || pixel < 0 || pixel > 1)
                throw new ArgumentException("Pixels must be finite and normalized between zero and one.", nameof(pixels));
        if (labels is not null)
            foreach (int label in labels)
                if ((uint)label >= ClassCount)
                    throw new ArgumentException("Digit labels must be between zero and nine.", nameof(labels));

        Pixels = pixels;
        Labels = labels;
        Count = pixels.Length / InputSize;
    }
}
