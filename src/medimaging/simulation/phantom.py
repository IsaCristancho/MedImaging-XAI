"""Synthetic phantoms used for medical imaging experiments."""

from __future__ import annotations

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def create_shepp_logan(size: int = 256) -> np.ndarray: 
    """Create a normalized Shepp-Logan phantom.

    Parameters
    ----------
    size:
        Output image size in pixels. The result is square.

    Returns
    -------
    np.ndarray
        A square float32 image with values approximately in [0, 1].

    Raises
    ------
    ValueError
        If size is not a positive integer.
    """
    if not isinstance(size, int) or isinstance(size, bool): 
        raise ValueError("size must be a positive integer")

    if size <= 0: 
        raise ValueError("size must be a positive integer")

    phantom = shepp_logan_phantom() 
    # Genera el phantom Shepp-Logan estándar (matriz 256x256)

    resized = resize(
        phantom,
        (size, size),
        anti_aliasing=True,
        preserve_range=True,
    ) # anti_aliasing=True suaviza el redimensionamiento 
    # preserve_range=True mantiene el rango de valores original [0,1]

    return resized.astype(np.float32) # Convierte a float32 (estándar en deep learning)
