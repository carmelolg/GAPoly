"""Rendering helpers on top of the numba kernels."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .kinds import POLYGON, Kind
from .shapes import Genome

Array = NDArray[np.float32]


def render(genome: Genome, width: int, height: int, background: tuple[float, float, float],
           kind: Kind = POLYGON) -> Array:
    """RGB float32 image in [0, 1], shape (height, width, 3)."""
    return kind.render_image(genome, width, height, np.asarray(background, dtype=np.float32))


def to_image(array: Array) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")
