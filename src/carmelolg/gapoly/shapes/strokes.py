"""Brush strokes: genome layout, rasteriser and operators.

A genome is a float32 array of shape ``(strokes, COLS)``, one row per stroke, drawn in row order (later rows on top).
Every value is in [0, 1]; coordinates are fractions of the image width and height, ``width`` is the thickness of the
brush as a fraction of the longest image side, so a genome renders at any resolution.

Layout of one row (11 columns)::

    column:   0   1 |  2   3 |  4   5 |   6   | 7  8  9 |  10
    meaning:  x0  y0 | cx  cy | x1  y1 | width | r  g  b | alpha

A stroke is a quadratic Bezier curve from ``(x0, y0)`` to ``(x1, y1)`` bent towards the control point ``(cx, cy)``,
painted with a round brush: every pixel closer than half the width to the curve is covered, with round caps at both
ends. The stroke is one flat colour and one opacity, so overlapping parts do not double up.

The rasteriser flattens the curve into short segments (about three pixels each) and measures the distance from the
pixel centre to the nearest segment.
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

X0, Y0, CX, CY, X1, Y1, WIDTH = 0, 1, 2, 3, 4, 5, 6
RED, GREEN, BLUE, ALPHA = 7, 8, 9, 10
COLS = 11

_MIN_HALF_PX = 0.7072  # a brush always reaches the centre of the pixels it passes over
_MAX_SEGMENTS = 32


def make_row(start: tuple[float, float], control: tuple[float, float], end: tuple[float, float], width: float,
             rgba: tuple[float, float, float, float]) -> Row:
    """One stroke row from its start, control point, end, brush width and colour ``(r, g, b, a)``."""
    return np.array([*start, *control, *end, width, *rgba], dtype=np.float32)


def extent(row: Row) -> float:
    """Largest side of the box around the three points of the stroke (the curve stays inside it)."""
    xs, ys = row[[X0, CX, X1]], row[[Y0, CY, Y1]]
    return float(max(xs.max() - xs.min(), ys.max() - ys.min()))


def describe(genome: Genome) -> str:
    """One line per stroke, in drawing order."""
    return "\n".join(
        f"#{k} stroke ({row[X0]:.2f},{row[Y0]:.2f}) via ({row[CX]:.2f},{row[CY]:.2f}) to ({row[X1]:.2f},{row[Y1]:.2f}) "
        f"width {row[WIDTH]:.3f} rgba({row[RED]:.2f},{row[GREEN]:.2f},{row[BLUE]:.2f},{row[ALPHA]:.2f})"
        for k, row in enumerate(genome))


def _width_bounds(limits: Limits) -> tuple[float, float]:
    low = max(limits.min_size, 1e-4)
    return low, max(low, limits.max_size / 3)  # the brush is thin compared with the stroke it paints


def random_row(rng: Random, limits: Limits, rgb: tuple[float, float, float] | None = None,
               center: tuple[float, float] | None = None) -> Row:
    """Random slightly curved stroke centred on ``center`` (random if omitted), at most ``max_size`` long."""
    low, high = _width_bounds(limits)
    cx, cy = center if center is not None else (rng.random(), rng.random())
    rgb = rgb if rgb is not None else (rng.random(), rng.random(), rng.random())
    length = rng.uniform(0.3, 1.0) * limits.max_size
    angle = rng.uniform(0, math.pi)
    dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
    bend = rng.uniform(-0.3, 0.3) * length  # sideways offset of the control point
    start = (clamp(cx - dx), clamp(cy - dy))
    end = (clamp(cx + dx), clamp(cy + dy))
    control = (clamp(cx - dy / (length / 2 or 1) * bend), clamp(cy + dx / (length / 2 or 1) * bend))
    alpha = rng.uniform(max(limits.min_alpha, 0.1), min(limits.max_alpha, 0.6))
    return make_row(start, control, end, rng.uniform(low, high), (*rgb, alpha))


def move_point(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.015) -> None:
    """Move the start, the control point or the end in place; undone if the stroke would grow past the cap."""
    before, old = row.copy(), extent(row)
    column = rng.choice((X0, CX, X1))
    row[column] = clamp(float(row[column]) + rng.gauss(0, sigma))
    row[column + 1] = clamp(float(row[column + 1]) + rng.gauss(0, sigma))
    if extent(row) > max(limits.max_size, old):  # never grow past the cap
        row[:] = before


def shift_stroke(row: np.ndarray, rng: Random, sigma: float = 0.02) -> None:
    dx, dy = rng.gauss(0, sigma), rng.gauss(0, sigma)
    for column in (X0, CX, X1):
        row[column] = clamp(float(row[column]) + dx)
        row[column + 1] = clamp(float(row[column + 1]) + dy)


def resize(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.25) -> None:
    """Scale the brush width by a log-normal factor; never grows past the cap (wider than it can only shrink)."""
    low, high = _width_bounds(limits)
    row[WIDTH] = clamp(float(row[WIDTH]) * math.exp(rng.gauss(0, sigma)), low, max(high, float(row[WIDTH])))


def change_color(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.06) -> None:
    for channel in (RED, GREEN, BLUE):
        row[channel] = clamp(float(row[channel]) + rng.gauss(0, sigma))
    row[ALPHA] = clamp(float(row[ALPHA]) + rng.gauss(0, sigma), limits.min_alpha, limits.max_alpha)


def _edit_shape(row: np.ndarray, rng: Random, limits: Limits) -> None:
    pick = rng.random()
    if pick < 0.5:
        move_point(row, rng, limits)
    elif pick < 0.75:
        shift_stroke(row, rng)
    else:
        resize(row, rng, limits)


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_stroke: float | None = None) -> Genome:
    """Bend, move or thicken some strokes, recolour some, add / remove / reorder (see ``operators.mutate_rows``)."""
    return mutate_rows(genome, rng, rate, limits, per_stroke, _edit_shape, change_color,
                       lambda r, lim: random_row(r, lim))


@njit(cache=True, nogil=True, fastmath=True)
def _curve(row, width, height, pts_x, pts_y):  # type: ignore[no-untyped-def]
    """Flatten the stroke into ``pts_x`` / ``pts_y``; returns the number of segments, the half brush width and the
    pixels ``(x0, x1, y0, y1)`` the stroke can paint."""
    side = max(width, height)
    ax, ay = row[X0] * width, row[Y0] * height
    bx, by = row[CX] * width, row[CY] * height
    ex, ey = row[X1] * width, row[Y1] * height
    half = max(row[WIDTH] * side / 2, _MIN_HALF_PX)
    length = np.hypot(bx - ax, by - ay) + np.hypot(ex - bx, ey - by)  # the control polygon bounds the curve length
    n = min(_MAX_SEGMENTS, max(4, int(length / 3.0)))
    min_x, max_x, min_y, max_y = width, 0.0, height, 0.0
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        pts_x[i] = u * u * ax + 2.0 * u * t * bx + t * t * ex
        pts_y[i] = u * u * ay + 2.0 * u * t * by + t * t * ey
        min_x, max_x = min(min_x, pts_x[i]), max(max_x, pts_x[i])
        min_y, max_y = min(min_y, pts_y[i]), max(max_y, pts_y[i])
    return (n, half, max(0, int(np.floor(min_x - half))), min(width, int(np.ceil(max_x + half)) + 1),
            max(0, int(np.floor(min_y - half))), min(height, int(np.ceil(max_y + half)) + 1))


@njit(cache=True, nogil=True, fastmath=True)
def _draw(genome, canvas, ox, oy, width, height):  # type: ignore[no-untyped-def]
    """Paint the strokes on ``canvas``, a window of a ``width`` x ``height`` image with its corner at ``(ox, oy)``."""
    pts_x = np.empty(_MAX_SEGMENTS + 1, dtype=np.float64)
    pts_y = np.empty(_MAX_SEGMENTS + 1, dtype=np.float64)
    side = max(width, height)
    for k in range(genome.shape[0]):
        # The curve stays inside the box of its three points: skip, without flattening it, the strokes that cannot
        # reach this window (the margin covers the brush and the rounding of the pixel box).
        reach = max(genome[k, WIDTH] * side / 2, _MIN_HALF_PX) + 3.0
        if (max(genome[k, X0], genome[k, CX], genome[k, X1]) * width + reach <= ox
                or min(genome[k, X0], genome[k, CX], genome[k, X1]) * width - reach >= ox + canvas.shape[1]
                or max(genome[k, Y0], genome[k, CY], genome[k, Y1]) * height + reach <= oy
                or min(genome[k, Y0], genome[k, CY], genome[k, Y1]) * height - reach >= oy + canvas.shape[0]):
            continue
        n, half, x0, x1, y0, y1 = _curve(genome[k], width, height, pts_x, pts_y)
        x0, x1 = max(x0, ox), min(x1, ox + canvas.shape[1])
        y0, y1 = max(y0, oy), min(y1, oy + canvas.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        alpha = genome[k, ALPHA]
        keep = 1.0 - alpha
        r = genome[k, RED] * alpha
        g = genome[k, GREEN] * alpha
        b = genome[k, BLUE] * alpha
        limit = half * half
        for py in range(y0, y1):
            yc = py + 0.5
            for px in range(x0, x1):
                xc = px + 0.5
                best = 1e300
                for i in range(n):  # squared distance from the pixel centre to segment i
                    sx, sy = pts_x[i], pts_y[i]
                    vx, vy = pts_x[i + 1] - sx, pts_y[i + 1] - sy
                    span = vx * vx + vy * vy
                    t = 0.0 if span == 0.0 else min(1.0, max(0.0, ((xc - sx) * vx + (yc - sy) * vy) / span))
                    dx, dy = xc - (sx + t * vx), yc - (sy + t * vy)
                    best = min(best, dx * dx + dy * dy)
                if best <= limit:
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
    pts_x = np.empty(_MAX_SEGMENTS + 1, dtype=np.float64)
    pts_y = np.empty(_MAX_SEGMENTS + 1, dtype=np.float64)
    out = np.empty((rows.shape[0], 4), dtype=np.int64)
    for k in range(rows.shape[0]):
        _, _, out[k, 0], out[k, 1], out[k, 2], out[k, 3] = _curve(rows[k], width, height, pts_x, pts_y)
    return out


def render_image(genome: np.ndarray, width: int, height: int, background: np.ndarray) -> np.ndarray:
    canvas = np.empty((height, width, 3), dtype=np.float32)
    render_into(np.ascontiguousarray(genome, dtype=np.float32), background, canvas)
    return canvas


class StrokeShape(Shape):
    name = "stroke"
    plural = "strokes"
    cols = COLS
    uses_size = True
    default_elements = 1500
    default_initial = 100

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
