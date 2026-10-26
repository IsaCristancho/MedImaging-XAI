
"""Combined image quality evaluation."""

import numpy as np

from medimaging.quality.mae import calculate_mae
from medimaging.quality.psnr import calculate_psnr
from medimaging.quality.rmse import calculate_rmse
from medimaging.quality.ssim import calculate_ssim


def evaluate_image_quality(
    reference: np.ndarray,
    image: np.ndarray,
) -> dict[str, float]:
    """Calculate four image quality metrics.

    Parameters
    ----------
    reference:
        Reference image used as ground truth.
    image:
        Image to evaluate, such as a CT reconstruction.

    Returns
    -------
    dict[str, float]
        PSNR, SSIM, MAE and RMSE values.
    """
    return {
        "psnr": calculate_psnr(reference, image),
        "ssim": calculate_ssim(reference, image),
        "mae": calculate_mae(reference, image),
        "rmse": calculate_rmse(reference, image),
    }
