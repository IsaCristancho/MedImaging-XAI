
"""Structural similarity for image quality evaluation."""

import numpy as np
from skimage.metrics import structural_similarity


def calculate_ssim(
    reference: np.ndarray,
    image: np.ndarray,
) -> float:
    """Calculate SSIM between a reference image and a reconstructed image."""
    if reference.shape != image.shape:
        raise ValueError("Images must have the same shape.")

    if reference.size == 0:
        raise ValueError("Images must not be empty.")

    if not np.isfinite(reference).all() or not np.isfinite(image).all():
        raise ValueError("Images must contain only finite values.")

    data_range = float(reference.max() - reference.min())

    if data_range <= 0:
        raise ValueError("Reference image must have a positive data range.")

    value = structural_similarity(
        reference,
        image,
        data_range=data_range,
    )

    return float(value)
