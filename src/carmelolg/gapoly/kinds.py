"""The kinds of element an image can be made of. Each kind brings its own genome layout, rasteriser and operators;
everything else (problem, viewer, command line) works on any kind through this table."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from typing import Any

import numpy as np

from . import fast_render, operators, shapes
from .limits import Limits


@dataclass(frozen=True)
class Kind:
    name: str  # value of ``--shape``
    singular: str  # "polygon", for messages
    plural: str
    cols: int  # columns of one genome row
    uses_size: bool  # whether ``Limits.max_size`` (and the size schedule) mean something for this kind
    default_elements: int  # default for ``--elements``
    default_initial: int  # default for ``--initial-elements``
    render_mse: Callable[..., float]  # (genome, target, background) -> mse, numba kernel
    render_image: Callable[..., Any]  # (genome, width, height, background) -> float32 image
    random_row: Callable[..., Any]  # (rng, limits, rgb, center) -> one row
    mutate: Callable[[Any, Random, float, Limits, float | None], Any]  # (genome, rng, rate, limits, per_element)
    describe: Callable[[Any], str]

    def empty_genome(self) -> np.ndarray:
        return np.zeros((0, self.cols), dtype=np.float32)


POLYGON = Kind("polygon", "polygon", "polygons", shapes.COLS, True, 400, 60, fast_render.render_mse,
               fast_render.render_image, shapes.random_row, operators.mutate, shapes.describe)

KINDS: dict[str, Kind] = {POLYGON.name: POLYGON}
