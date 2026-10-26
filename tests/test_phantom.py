"""Tests for synthetic medical imaging phantoms."""

import numpy as np
import pytest

from medimaging.simulation.phantom import create_shepp_logan


def test_create_shepp_logan_default_shape():
    """The default phantom should be a 256x256 image."""
    phantom = create_shepp_logan()

    assert phantom.shape == (256, 256)


def test_create_shepp_logan_custom_size():
    """The phantom should respect the requested size."""
    phantom = create_shepp_logan(128)

    assert phantom.shape == (128, 128)


def test_create_shepp_logan_dtype():
    """The phantom should use float32 values."""
    phantom = create_shepp_logan()

    assert phantom.dtype == np.float32


def test_create_shepp_logan_value_range():
    """The phantom values should be approximately normalized to [0, 1]."""
    phantom = create_shepp_logan()

    assert phantom.min() >= 0.0
    assert phantom.max() <= 1.0


@pytest.mark.parametrize("invalid_size", [0, -1, -256, 2.5, "256", True])
def test_create_shepp_logan_rejects_invalid_size(invalid_size):
    """Invalid image sizes should raise ValueError."""
    with pytest.raises(ValueError, match="size must be a positive integer"):
        create_shepp_logan(invalid_size)