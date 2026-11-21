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
from ..core.kernels import canvas_mse, fill, merge_boxes, unmatched_rows, window_error
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
def _box(row, width, height):  # type: ignore[no-untyped-def]
    """Pixels ``(x0, x1, y0, y1)`` a disc row can paint."""
    radius = max(row[RADIUS] * max(width, height), _MIN_RADIUS_PX)
    cx, cy = row[X] * width, row[Y] * height
    return (max(0, int(np.floor(cx - radius))), min(width, int(np.ceil(cx + radius)) + 1),
            max(0, int(np.floor(cy - radius))), min(height, int(np.ceil(cy + radius)) + 1))


@njit(cache=True, nogil=True, fastmath=True)
def _draw(genome, canvas, ox, oy, width, height):  # type: ignore[no-untyped-def]
    """Paint the discs on ``canvas``, a window of a ``width`` x ``height`` image with its corner at ``(ox, oy)``."""
    side = max(width, height)
    for k in range(genome.shape[0]):
        x0, x1, y0, y1 = _box(genome[k], width, height)
        x0, x1 = max(x0, ox), min(x1, ox + canvas.shape[1])
        y0, y1 = max(y0, oy), min(y1, oy + canvas.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        cx = genome[k, X] * width
        cy = genome[k, Y] * height
        radius = max(genome[k, RADIUS] * side, _MIN_RADIUS_PX)
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
                    canvas[py - oy, px - ox, 0] = canvas[py - oy, px - ox, 0] * keep + r
                    canvas[py - oy, px - ox, 1] = canvas[py - oy, px - ox, 1] * keep + g
                    canvas[py - oy, px - ox, 2] = canvas[py - oy, px - ox, 2] * keep + b

@njit(cache=True, nogil=True, fastmath=True)
def render_into(genome, background, canvas):  # type: ignore[no-untyped-def]
    fill(canvas, background)
    _draw(genome, canvas, 0, 0, canvas.shape[1], canvas.shape[0])


@njit(cache=True, nogil=True, fastmath=True)
def render_mse(genome, target, background):  # type: ignore[no-untyped-def]
    canvas = np.empty_like(target)
    render_into(genome, background, canvas)
    return canvas_mse(canvas, target)


@njit(cache=True, nogil=True, fastmath=True)
def window_sse(genome, target, background, x0, y0, x1, y1):  # type: ignore[no-untyped-def]
    """Squared error over the window ``x0 <= x < x1``, ``y0 <= y < y1``: only that window is drawn."""
    canvas = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
    fill(canvas, background)
    _draw(genome, canvas, x0, y0, target.shape[1], target.shape[0])
    return window_error(canvas, target, x0, y0)


@njit(cache=True, nogil=True, fastmath=True)
def boxes(rows, width, height):  # type: ignore[no-untyped-def]
    out = np.empty((rows.shape[0], 4), dtype=np.int64)
    for k in range(rows.shape[0]):
        out[k, 0], out[k, 1], out[k, 2], out[k, 3] = _box(rows[k], width, height)
    return out


@njit(cache=True, nogil=True)  # no fastmath: NaN means "draw in full"
def delta_sse(before, after, target, background, max_rows, max_pixels):  # type: ignore[no-untyped-def]
    """Squared error of ``after`` minus that of ``before`` (two genomes that differ in a few rows), drawing only the
    windows around the rows that differ. NaN when it is not worth it: more than ``max_rows`` rows differ or the
    windows cover more than ``max_pixels``. Everything runs in compiled code that releases the GIL."""
    height, width = target.shape[0], target.shape[1]
    old_rows, new_rows, n_old, n_new = unmatched_rows(before, after, max_rows)
    if n_old < 0:
        return np.nan
    if n_old == 0 and n_new == 0:
        return 0.0
    areas = np.empty((n_old + n_new, 4), dtype=np.int64)
    for k in range(n_old + n_new):
        at = k
        row = before[old_rows[k]] if k < n_old else after[new_rows[k - n_old]]
        areas[at, 0], areas[at, 1], areas[at, 2], areas[at, 3] = _box(row, width, height)
    windows = merge_boxes(areas)
    pixels = 0
    for w in range(windows.shape[0]):
        pixels += (windows[w, 1] - windows[w, 0]) * (windows[w, 3] - windows[w, 2])
    if pixels > max_pixels:
        return np.nan
    delta = 0.0
    for w in range(windows.shape[0]):
        x0, x1, y0, y1 = windows[w, 0], windows[w, 1], windows[w, 2], windows[w, 3]
        delta += window_sse(after, target, background, x0, y0, x1, y1) - window_sse(before, target, background, x0, y0, x1, y1)
    return delta


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

    def window_sse(self, genome: Genome, target: Array, background: Array, x0: int, y0: int, x1: int,
                   y1: int) -> float:
        return float(window_sse(genome, target, background, x0, y0, x1, y1))

    def boxes(self, rows: Genome, width: int, height: int) -> np.ndarray:
        return boxes(rows, width, height)

    def delta_sse(self, before: Genome, after: Genome, target: Array, background: Array, max_rows: int,
                  max_pixels: int) -> float:
        return float(delta_sse(before, after, target, background, max_rows, max_pixels))

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
