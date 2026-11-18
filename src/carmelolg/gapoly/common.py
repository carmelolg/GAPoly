"""Types and limits shared by every shape."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Genome = NDArray[np.float32]  # (elements, columns), one row per element
Row = NDArray[np.float32]  # one element
Array = NDArray[np.float32]  # an RGB image in [0, 1], shape (height, width, 3)


@dataclass(frozen=True)
class Limits:
    """Bounds that mutation and crossover keep.

    ``max_size`` is the largest extent of one element as a fraction of the image side: the width or height of a
    polygon, the diameter of a disc (Voronoi points have no size). ``min_size`` is the smallest diameter of a disc.
    The vertex limits only matter for polygons, the alpha limits for polygons and discs.
    """

    min_elements: int = 1
    max_elements: int = 150
    min_vertices: int = 3
    max_vertices: int = 6
    min_alpha: float = 0.05
    max_alpha: float = 0.95
    max_size: float = 0.12
    min_size: float = 0.0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else min(value, high)
