
"""Tests for image quality metrics."""

import numpy as np
import pytest

from medimaging.quality.evaluation import evaluate_image_quality
from medimaging.quality.mae import calculate_mae
from medimaging.quality.psnr import calculate_psnr
from medimaging.quality.rmse import calculate_rmse
from medimaging.quality.ssim import calculate_ssim


@pytest.fixture
def reference_image():
    """Create a small synthetic reference image."""
    image = np.zeros((64, 64), dtype=np.float32)
    image[16:48, 16:48] = 1.0
    return image


def test_psnr_identical_images_is_infinite(reference_image):
    """Identical images should have infinite PSNR."""
    assert np.isinf(calculate_psnr(reference_image, reference_image))


def test_ssim_identical_images_is_one(reference_image):
    """Identical images should have SSIM equal to one."""
    assert calculate_ssim(reference_image, reference_image) == pytest.approx(1.0)


def test_psnr_decreases_with_noise(reference_image):
    """Adding noise should reduce PSNR."""
    noisy_image = reference_image.copy()
    noisy_image[20:24, 20:24] = 0.5

    assert calculate_psnr(reference_image, noisy_image) < float("inf")


def test_ssim_decreases_with_noise(reference_image):
    """Adding noise should reduce SSIM."""
    noisy_image = reference_image.copy()
    noisy_image[20:24, 20:24] = 0.5

    assert calculate_ssim(reference_image, noisy_image) < 1.0


@pytest.mark.parametrize(
    "metric",
    [calculate_psnr, calculate_ssim],
)
def test_metrics_reject_different_shapes(metric):
    """Metrics should reject images with different shapes."""
    reference = np.zeros((32, 32), dtype=np.float32)
    image = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="same shape"):
        metric(reference, image)


@pytest.mark.parametrize(
    "metric",
    [calculate_psnr, calculate_ssim],
)
def test_metrics_reject_empty_images(metric):
    """Metrics should reject empty images."""
    reference = np.empty((0, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="must not be empty"):
        metric(reference, reference)


@pytest.mark.parametrize(
    "metric",
    [calculate_psnr, calculate_ssim],
)
def test_metrics_reject_non_finite_values(metric):
    """Metrics should reject NaN and infinite values."""
    reference = np.zeros((32, 32), dtype=np.float32)
    image = reference.copy()
    image[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        metric(reference, image)


@pytest.mark.parametrize(
    "metric",
    [calculate_psnr, calculate_ssim],
)
def test_metrics_reject_constant_reference(metric):
    """Metrics should reject a reference with no intensity variation."""
    reference = np.ones((32, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="positive data range"):
        metric(reference, reference)


@pytest.mark.parametrize(
    "metric",
    [calculate_mae, calculate_rmse],
)
def test_error_metrics_return_zero_for_identical_images(
    metric,
    reference_image,
):
    """Error metrics should return zero for identical images."""
    assert metric(reference_image, reference_image) == pytest.approx(0.0)


def test_mae_calculates_absolute_error():
    """MAE should calculate the mean absolute difference."""
    reference = np.zeros((4, 4), dtype=np.float32)
    image = np.ones((4, 4), dtype=np.float32)

    assert calculate_mae(reference, image) == pytest.approx(1.0)


def test_rmse_calculates_squared_error():
    """RMSE should calculate the root mean squared difference."""
    reference = np.zeros((4, 4), dtype=np.float32)
    image = np.ones((4, 4), dtype=np.float32)

    assert calculate_rmse(reference, image) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "metric",
    [calculate_mae, calculate_rmse],
)
def test_error_metrics_reject_different_shapes(metric):
    """Error metrics should reject images with different shapes."""
    reference = np.zeros((32, 32), dtype=np.float32)
    image = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="same shape"):
        metric(reference, image)


def test_evaluate_image_quality_returns_all_metrics(reference_image):
    metrics = evaluate_image_quality(
        reference=reference_image,
        image=reference_image.copy(),
    )

    assert set(metrics) == {"psnr", "ssim", "mae", "rmse"}
    assert np.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_evaluate_image_quality_detects_shape_mismatch():
    reference = np.zeros((8, 8))
    image = np.zeros((16, 16))

    with pytest.raises(ValueError, match="same shape"):
        evaluate_image_quality(reference, image)



def test_evaluate_image_quality_rejects_empty_images():
    reference = np.empty((0, 0))
    image = np.empty((0, 0))

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_image_quality(reference, image)


def test_evaluate_image_quality_rejects_non_finite_values():
    reference = np.ones((8, 8))
    image = reference.copy()
    image[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        evaluate_image_quality(reference, image)

