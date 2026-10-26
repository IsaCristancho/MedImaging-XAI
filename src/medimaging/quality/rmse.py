
"""Root mean squared error for image quality evaluation."""

import numpy as np


def calculate_rmse(
    reference: np.ndarray,
    image: np.ndarray,
) -> float:
    """Calculate the root mean squared error between two images."""
    if reference.shape != image.shape:
        raise ValueError("Images must have the same shape.")

    if reference.size == 0:
        raise ValueError("Images must not be empty.")

    if not np.isfinite(reference).all() or not np.isfinite(image).all():
        raise ValueError("Images must contain only finite values.")

    return float(np.sqrt(np.mean((reference - image) ** 2)))
