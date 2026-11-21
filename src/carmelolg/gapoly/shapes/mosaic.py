"""Mosaic: hexagonal tiles on a honeycomb lattice, genome layout, rasteriser and operators.

A genome is a float32 array of shape ``(tiles, COLS)``, one row per tile, drawn in row order (later rows on top).
Every value is in [0, 1]; ``x`` and ``y`` are fractions of the image width and height, ``radius`` is a fraction of the
longest image side, so a genome renders at any resolution.

Layout of one row (6 columns)::

    column:   0   1 |    2   | 3  4  5
    meaning:  x   y | radius | r  g  b

A tile is an opaque, flat-coloured regular hexagon (flat top, ``radius`` is the distance from its centre to a vertex).
Tiles are not free: the rasteriser snaps every tile to a honeycomb lattice, so tiles of the same size fit together
without gaps or overlaps, like the tesserae of a mosaic.

- The radius is rounded to a size level, a step of a factor of the square root of two, counted from ``_R0``. Tiles
  that share a level share a lattice; big tiles cover a region at once, small ones add the detail.
- ``x`` and ``y`` only choose which lattice cell the tile sits in: the tile is drawn centred on the nearest lattice
  node, so moving a tile by less than half a cell does not change the image.

There is no alpha. With no tile the image is the background. A tile is never smaller than one pixel of radius.
"""
from __future__ import annotations

import math
from random import Random

import numpy as np
from numba import njit

from ..core.common import Array, Genome, Limits, Row, clamp
from ..core.kernels import canvas_mse, fill, window_error
from ..core.operators import mutate_rows
from ..core.shape import Shape

X, Y, RADIUS = 0, 1, 2
RED, GREEN, BLUE = 3, 4, 5
COLS = 6

_R0 = 2.0**-10  # radius of size level 0, as a fraction of the longest side; every level is sqrt(2) times the previous
_MIN_RADIUS_PX = 1.0  # a tile never shrinks below one pixel of radius
_SQRT3 = 1.7320508075688772


def make_row(x: float, y: float, radius: float, rgb: tuple[float, float, float]) -> Row:
    """One tile row from its centre, radius and colour ``(r, g, b)``."""
    return np.array([x, y, radius, *rgb], dtype=np.float32)


def describe(genome: Genome) -> str:
    """One line per tile, in drawing order."""
    return "\n".join(
        f"#{k} tile ({row[X]:.2f},{row[Y]:.2f}) radius {row[RADIUS]:.3f} "
        f"rgb({row[RED]:.2f},{row[GREEN]:.2f},{row[BLUE]:.2f})"
        for k, row in enumerate(genome))


@njit(cache=True, nogil=True, fastmath=True)
def quantise(radius):  # type: ignore[no-untyped-def]
    """The radius of the size level nearest to ``radius`` (both fractions of the longest side)."""
    return _R0 * 2.0 ** (np.round(2.0 * np.log2(radius / _R0)) / 2.0)


@njit(cache=True, nogil=True, fastmath=True)
def snap(cx, cy, radius):  # type: ignore[no-untyped-def]
    """Centre of the lattice hexagon (flat top, circumradius ``radius``) that contains the point ``(cx, cy)``.

    Lattice nodes are at ``(1.5 r i, sqrt(3) r (j + (i % 2) / 2))`` for whole ``i`` and ``j``. The nearest node
    is among the three columns around the point, one node per column.
    """
    step_x = 1.5 * radius
    step_y = _SQRT3 * radius
    first = int(np.floor(cx / step_x + 0.5))
    best = 1e300
    best_x = cx
    best_y = cy
    for i in range(first - 1, first + 2):
        offset = 0.5 * (i & 1)
        j = np.floor(cy / step_y - offset + 0.5)
        hx = i * step_x
        hy = (j + offset) * step_y
        d = (hx - cx) ** 2 + (hy - cy) ** 2
        if d < best:
            best = d
            best_x = hx
            best_y = hy
    return best_x, best_y


def _radius_bounds(limits: Limits) -> tuple[float, float]:
    low = max(limits.min_size / 2, 1e-4)
    return low, max(low, limits.max_size / 2)


def random_row(rng: Random, limits: Limits, rgb: tuple[float, float, float] | None = None,
               center: tuple[float, float] | None = None) -> Row:
    """Random tile at ``center`` (random if omitted), at most ``max_size`` wide. ``rgb`` is the colour."""
    low, high = _radius_bounds(limits)
    cx, cy = center if center is not None else (rng.random(), rng.random())
    rgb = rgb if rgb is not None else (rng.random(), rng.random(), rng.random())
    return make_row(cx, cy, rng.uniform(low, high), rgb)


def move(row: np.ndarray, rng: Random, limits: Limits) -> None:
    """Hop about one cell: the step follows the tile size, since a smaller move would not change the image."""
    sigma = max(0.01, 1.5 * float(row[RADIUS]))
    row[X] = clamp(float(row[X]) + rng.gauss(0, sigma))
    row[Y] = clamp(float(row[Y]) + rng.gauss(0, sigma))


def resize(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.35) -> None:
    """Scale the radius by a log-normal factor; never grows past the cap (a tile already above it can only shrink)."""
    low, high = _radius_bounds(limits)
    row[RADIUS] = clamp(float(row[RADIUS]) * math.exp(rng.gauss(0, sigma)), low, max(high, float(row[RADIUS])))


def change_color(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.06) -> None:
    for channel in (RED, GREEN, BLUE):
        row[channel] = clamp(float(row[channel]) + rng.gauss(0, sigma))


def _edit_shape(row: np.ndarray, rng: Random, limits: Limits) -> None:
    if rng.random() < 0.6:
        move(row, rng, limits)
    else:
        resize(row, rng, limits)


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_tile: float | None = None) -> Genome:
    """Move or resize some tiles, recolour some, add / remove / reorder (see ``operators.mutate_rows``)."""
    return mutate_rows(genome, rng, rate, limits, per_tile, _edit_shape, change_color,
                       lambda r, lim: random_row(r, lim))


@njit(cache=True, nogil=True, fastmath=True)
def _tile(row, width, height):  # type: ignore[no-untyped-def]
    """The tile of a row: radius and centre of its lattice hexagon in pixels, and the pixels ``(x0, x1, y0, y1)``
    it can paint."""
    radius = max(quantise(row[RADIUS]) * max(width, height), _MIN_RADIUS_PX)
    hx, hy = snap(row[X] * width, row[Y] * height, radius)
    half_height = 0.5 * _SQRT3 * radius
    return (radius, hx, hy, max(0, int(np.floor(hx - radius))), min(width, int(np.ceil(hx + radius)) + 1),
            max(0, int(np.floor(hy - half_height))), min(height, int(np.ceil(hy + half_height)) + 1))


@njit(cache=True, nogil=True, fastmath=True)
def _draw(genome, canvas, ox, oy, width, height):  # type: ignore[no-untyped-def]
    """Paint the tiles on ``canvas``, a window of a ``width`` x ``height`` image with its corner at ``(ox, oy)``."""
    for k in range(genome.shape[0]):
        radius, hx, hy, x0, x1, y0, y1 = _tile(genome[k], width, height)
        x0, x1 = max(x0, ox), min(x1, ox + canvas.shape[1])
        y0, y1 = max(y0, oy), min(y1, oy + canvas.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        half_height = 0.5 * _SQRT3 * radius
        r = genome[k, RED]
        g = genome[k, GREEN]
        b = genome[k, BLUE]
        for py in range(y0, y1):
            dy = abs(py + 0.5 - hy)
            if dy > half_height:
                continue
            reach = radius - dy / _SQRT3  # half the width of the hexagon at this height
            for px in range(x0, x1):
                if abs(px + 0.5 - hx) <= reach:
                    canvas[py - oy, px - ox, 0] = r
                    canvas[py - oy, px - ox, 1] = g
                    canvas[py - oy, px - ox, 2] = b

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
        _, _, _, out[k, 0], out[k, 1], out[k, 2], out[k, 3] = _tile(rows[k], width, height)
    return out


def render_image(genome: np.ndarray, width: int, height: int, background: np.ndarray) -> np.ndarray:
    canvas = np.empty((height, width, 3), dtype=np.float32)
    render_into(np.ascontiguousarray(genome, dtype=np.float32), background, canvas)
    return canvas


class MosaicShape(Shape):
    name = "mosaic"
    plural = "tiles"
    cols = COLS
    uses_size = True
    default_elements = 3000
    default_initial = 200

    def render_mse(self, genome: Genome, target: Array, background: Array) -> float:
        return float(render_mse(genome, target, background))

    def window_sse(self, genome: Genome, target: Array, background: Array, x0: int, y0: int, x1: int,
                   y1: int) -> float:
        return float(window_sse(genome, target, background, x0, y0, x1, y1))

    def boxes(self, rows: Genome, width: int, height: int) -> np.ndarray:
        return boxes(rows, width, height)

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
