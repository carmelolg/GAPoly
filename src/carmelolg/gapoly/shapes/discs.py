"""Discs: genome layout, rasteriser and operators.

A genome is a float32 array of shape ``(discs, COLS)``, one row per disc, drawn in row order (later rows on top).
Every value is in [0, 1]; ``x`` and ``y`` are fractions of the image width and height, ``radius`` is a fraction of the
longest image side, so a genome renders at any resolution and the discs stay round.

Layout of one row (7 columns)::

    column:   0   1 |    2   | 3  4  5 |  6
    meaning:  x   y | radius | r  g  b | alpha

A disc is never smaller than one pixel: the rasteriser draws at least the pixel under its centre. A disc of
one pixel is the "single pixel" point; a bigger radius covers a group of pixels.
"""
from __future__ import annotations

import math
from random import Random

import numpy as np
from numba import njit

from ..core.common import Array, Genome, Limits, Row, clamp
from ..core.kernels import canvas_mse, fill
from ..core.operators import mutate_rows
from ..core.shape import Shape

X, Y, RADIUS = 0, 1, 2
RED, GREEN, BLUE, ALPHA = 3, 4, 5, 6
COLS = 7

_MIN_RADIUS_PX = 0.7072  # a disc always reaches the centre of the pixel it sits in (half the pixel diagonal)


def make_row(x: float, y: float, radius: float, rgba: tuple[float, float, float, float]) -> Row:
    """One disc row from its centre, radius and colour ``(r, g, b, a)``."""
    return np.array([x, y, radius, *rgba], dtype=np.float32)


def describe(genome: Genome) -> str:
    """One line per disc, in drawing order."""
    return "\n".join(
        f"#{k} disc ({row[X]:.2f},{row[Y]:.2f}) radius {row[RADIUS]:.3f} "
        f"rgba({row[RED]:.2f},{row[GREEN]:.2f},{row[BLUE]:.2f},{row[ALPHA]:.2f})"
        for k, row in enumerate(genome))


def random_row(rng: Random, limits: Limits, rgb: tuple[float, float, float] | None = None,
               center: tuple[float, float] | None = None) -> Row:
    """Random disc at ``center`` (random if omitted), at most ``max_size`` wide. ``rgb`` is the colour."""
    low, high = _radius_bounds(limits)
    cx, cy = center if center is not None else (rng.random(), rng.random())
    rgb = rgb if rgb is not None else (rng.random(), rng.random(), rng.random())
    alpha = rng.uniform(max(limits.min_alpha, 0.1), min(limits.max_alpha, 0.6))
    return make_row(cx, cy, rng.uniform(low, high), (*rgb, alpha))


def _radius_bounds(limits: Limits) -> tuple[float, float]:
    low = max(limits.min_size / 2, 1e-4)
    return low, max(low, limits.max_size / 2)


def move(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.015) -> None:
    row[X] = clamp(float(row[X]) + rng.gauss(0, sigma))
    row[Y] = clamp(float(row[Y]) + rng.gauss(0, sigma))


def resize(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.25) -> None:
    """Scale the radius by a log-normal factor; never grows past the cap (a disc already above it can only shrink)."""
    low, high = _radius_bounds(limits)
    row[RADIUS] = clamp(float(row[RADIUS]) * math.exp(rng.gauss(0, sigma)), low, max(high, float(row[RADIUS])))


def change_color(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.06) -> None:
    for channel in (RED, GREEN, BLUE):
        row[channel] = clamp(float(row[channel]) + rng.gauss(0, sigma))
    row[ALPHA] = clamp(float(row[ALPHA]) + rng.gauss(0, sigma), limits.min_alpha, limits.max_alpha)


def _edit_shape(row: np.ndarray, rng: Random, limits: Limits) -> None:
    if rng.random() < 0.5:
        move(row, rng, limits)
    else:
        resize(row, rng, limits)


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_disc: float | None = None) -> Genome:
    """Move or resize some discs, recolour some, add / remove / reorder (see ``operators.mutate_rows``)."""
    return mutate_rows(genome, rng, rate, limits, per_disc, _edit_shape, change_color,
                       lambda r, lim: random_row(r, lim))


@njit(cache=True, nogil=True, fastmath=True)
def _draw(genome, canvas):  # type: ignore[no-untyped-def]
    height, width = canvas.shape[0], canvas.shape[1]
    side = max(width, height)
    for k in range(genome.shape[0]):
        cx = genome[k, X] * width
        cy = genome[k, Y] * height
        radius = max(genome[k, RADIUS] * side, _MIN_RADIUS_PX)
        x0, x1 = max(0, int(np.floor(cx - radius))), min(width, int(np.ceil(cx + radius)) + 1)
        y0, y1 = max(0, int(np.floor(cy - radius))), min(height, int(np.ceil(cy + radius)) + 1)
        alpha = genome[k, ALPHA]
        keep = 1.0 - alpha
        r = genome[k, RED] * alpha
        g = genome[k, GREEN] * alpha
        b = genome[k, BLUE] * alpha
        limit = radius * radius
        for py in range(y0, y1):
            dy = py + 0.5 - cy
            for px in range(x0, x1):
                dx = px + 0.5 - cx
                if dx * dx + dy * dy <= limit:
                    canvas[py, px, 0] = canvas[py, px, 0] * keep + r
                    canvas[py, px, 1] = canvas[py, px, 1] * keep + g
                    canvas[py, px, 2] = canvas[py, px, 2] * keep + b


@njit(cache=True, nogil=True, fastmath=True)
def render_into(genome, background, canvas):  # type: ignore[no-untyped-def]
    fill(canvas, background)
    _draw(genome, canvas)


@njit(cache=True, nogil=True, fastmath=True)
def render_mse(genome, target, background):  # type: ignore[no-untyped-def]
    canvas = np.empty_like(target)
    render_into(genome, background, canvas)
    return canvas_mse(canvas, target)


def render_image(genome: np.ndarray, width: int, height: int, background: np.ndarray) -> np.ndarray:
    canvas = np.empty((height, width, 3), dtype=np.float32)
    render_into(np.ascontiguousarray(genome, dtype=np.float32), background, canvas)
    return canvas


class DiscShape(Shape):
    name = "disc"
    plural = "discs"
    cols = COLS
    uses_size = True
    default_elements = 2000
    default_initial = 200

    def render_mse(self, genome: Genome, target: Array, background: Array) -> float:
        return float(render_mse(genome, target, background))

    def render_image(self, genome: Genome, width: int, height: int, background: Array) -> Array:
        return render_image(genome, width, height, background)

    def random_row(self, rng: Random, limits: Limits, rgb: tuple[float, float, float] | None = None,
                   center: tuple[float, float] | None = None) -> Row:
        return random_row(rng, limits, rgb, center)

    def mutate(self, genome: Genome, rng: Random, rate: float, limits: Limits,
               per_element: float | None = None) -> Genome:
        return mutate(genome, rng, rate, limits, per_element)

    def describe(self, genome: Genome) -> str:
        return describe(genome)
