import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class HistEqNormalize(mcolors.Normalize):
    """
    Matplotlib normalization class for histogram equalization.
    Automatically calculates the mapping (ECDF) on the first image it sees.
    """
    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self._reference_sorted = None
        self._reference_cdf = None

    def __call__(self, value, clip=None):
        # Convert input to a masked array and check if it's a scalar
        value, is_scalar = self.process_value(value)

        # Mask out any non-finite values (NaNs, Infs)
        value = np.ma.masked_invalid(value)

        if self._reference_sorted is None:
            # Extract valid pixels to define the reference distribution
            valid_data = value.compressed()

            if valid_data.size == 0:
                return np.ma.masked_array(np.zeros_like(value))

            sorted_data = np.sort(valid_data)
            cdf = np.linspace(0.0, 1.0, sorted_data.size)

            # np.interp requires *strictly increasing* x-values.
            # Astronomical images often have large regions of nearly
            # identical pixel values, producing long runs of duplicates
            # in the sorted array.  We keep the *last* occurrence of
            # each value, which gives P(X <= x) — the correct quantile.
            # Keep the *first* occurrence of each run so that
            # the minimum pixel value maps to 0 (rather than to
            # the fraction of zero-valued background pixels).
            keep = np.concatenate(
                [[True], sorted_data[1:] != sorted_data[:-1]]
            )
            self._reference_sorted = sorted_data[keep]
            self._reference_cdf = cdf[keep]

        # Map values via linear interpolation on the ECDF.
        # reference_sorted is strictly increasing (no duplicates), so
        # np.interp is safe and fast.
        result = np.interp(value.data, self._reference_sorted, self._reference_cdf)

        # Restore the mask
        if np.ma.is_masked(value):
            result = np.ma.array(result, mask=value.mask)

        if is_scalar:
            result = result[0]

        return result

    def inverse(self, value):
        """Map from [0, 1] back to the original data range."""
        if self._reference_sorted is None:
            raise ValueError("Normalization uninitialized. Pass data through __call__ first.")

        value, is_scalar = self.process_value(value)
        value = np.ma.masked_invalid(value)

        result = np.interp(value.data, self._reference_cdf, self._reference_sorted)

        if np.ma.is_masked(value):
            result = np.ma.array(result, mask=value.mask)

        if is_scalar:
            result = result[0]

        return result


def main():
    # Generate mock astronomical data with a heavily skewed distribution
    np.random.seed(42)
    image_data = np.random.lognormal(mean=0, sigma=2, size=(500, 500))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Standard linear normalization
    im1 = axs[0, 0].imshow(image_data, cmap='viridis')
    axs[0, 0].set_title("Standard Linear Scale")
    fig.colorbar(im1, ax=axs[0, 0])

    axs[0, 1].hist(image_data.ravel(), bins=100, range=(0, 50))
    axs[0, 1].set_title("Original Histogram (Zoomed In)")

    # 2. Custom Histogram Equalization
    norm = HistEqNormalize()
    mapped_data = norm(image_data)
    print(np.percentile(mapped_data, [0., 10., 50., 90., 100.]))
    im2 = axs[1, 0].imshow(image_data, cmap='viridis', norm=norm)
    axs[1, 0].set_title("Histogram Equalized Scale")
    fig.colorbar(im2, ax=axs[1, 0])

    # Plot the flattened histogram of the mapped values
    # mapped_data = norm(image_data)
    axs[1, 1].hist(mapped_data.flatten(), bins=100, range=(0, 1))
    axs[1, 1].set_title("Equalized Histogram")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()