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

from random import Random

import numpy as np
from numba import njit

from ..core.common import Array, Genome, Limits, Row
from ..core.kernels import (
    L_MAX_ALPHA,
    L_MAX_ELEMENTS,
    L_MAX_SIZE,
    L_MIN_ALPHA,
    L_MIN_SIZE,
    canvas_mse,
    clamp,
    drop_and_move,
    fill,
    insert_row,
    limits_array,
    merge_boxes,
    pick,
    unmatched_rows,
    window_error,
)
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


@njit(cache=True, nogil=True, fastmath=True)
def _extent(row):  # type: ignore[no-untyped-def]
    """Largest side of the box around the three points of the stroke (the curve stays inside it)."""
    xs = np.array([row[X0], row[CX], row[X1]])
    ys = np.array([row[Y0], row[CY], row[Y1]])
    return max(xs.max() - xs.min(), ys.max() - ys.min())


@njit(cache=True, nogil=True, fastmath=True)
def _width_bounds(lim):  # type: ignore[no-untyped-def]
    low = max(lim[L_MIN_SIZE], 1e-4)
    return low, max(low, lim[L_MAX_SIZE] / 3)  # the brush is thin compared with the stroke it paints


@njit(cache=True, nogil=True, fastmath=True)
def _random_row(lim, cx, cy, r, g, b):  # type: ignore[no-untyped-def]
    """Random slightly curved stroke centred on ``(cx, cy)``, at most ``max_size`` long; negative values are random."""
    low, high = _width_bounds(lim)
    if cx < 0.0:
        cx, cy = np.random.random(), np.random.random()
    if r < 0.0:
        r, g, b = np.random.random(), np.random.random(), np.random.random()
    length = np.random.uniform(0.3, 1.0) * lim[L_MAX_SIZE]
    angle = np.random.uniform(0.0, np.pi)
    dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
    bend = np.random.uniform(-0.3, 0.3) * length  # sideways offset of the control point
    half = length / 2
    if half == 0.0:
        half = 1.0
    row = np.empty(COLS, dtype=np.float32)
    row[X0], row[Y0] = clamp(cx - dx, 0.0, 1.0), clamp(cy - dy, 0.0, 1.0)
    row[X1], row[Y1] = clamp(cx + dx, 0.0, 1.0), clamp(cy + dy, 0.0, 1.0)
    row[CX], row[CY] = clamp(cx - dy / half * bend, 0.0, 1.0), clamp(cy + dx / half * bend, 0.0, 1.0)
    row[RED], row[GREEN], row[BLUE] = r, g, b
    row[ALPHA] = np.random.uniform(max(lim[L_MIN_ALPHA], 0.1), min(lim[L_MAX_ALPHA], 0.6))
    row[WIDTH] = np.random.uniform(low, high)
    return row


@njit(cache=True, nogil=True, fastmath=True)
def _new_row(lim):  # type: ignore[no-untyped-def]
    return _random_row(lim, -1.0, -1.0, -1.0, -1.0, -1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _move_point(row, lim):  # type: ignore[no-untyped-def]
    """Move the start, the control point or the end in place; undone if the stroke would grow past the cap."""
    before = row.copy()
    old = _extent(row)
    column = 2 * np.random.randint(0, 3)  # X0, CX or X1
    row[column] = clamp(row[column] + np.random.normal(0.0, 0.015), 0.0, 1.0)
    row[column + 1] = clamp(row[column + 1] + np.random.normal(0.0, 0.015), 0.0, 1.0)
    if _extent(row) > max(lim[L_MAX_SIZE], old):  # never grow past the cap
        row[:] = before


@njit(cache=True, nogil=True, fastmath=True)
def _shift_stroke(row, lim):  # type: ignore[no-untyped-def]
    dx, dy = np.random.normal(0.0, 0.02), np.random.normal(0.0, 0.02)
    for column in (X0, CX, X1):
        row[column] = clamp(row[column] + dx, 0.0, 1.0)
        row[column + 1] = clamp(row[column + 1] + dy, 0.0, 1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _resize(row, lim):  # type: ignore[no-untyped-def]
    """Scale the brush width by a log-normal factor; never grows past the cap (wider than it can only shrink)."""
    low, high = _width_bounds(lim)
    row[WIDTH] = clamp(row[WIDTH] * np.exp(np.random.normal(0.0, 0.25)), low, max(high, row[WIDTH]))


@njit(cache=True, nogil=True, fastmath=True)
def _edit_shape(row, lim):  # type: ignore[no-untyped-def]
    pick = np.random.random()
    if pick < 0.5:
        _move_point(row, lim)
    elif pick < 0.75:
        _shift_stroke(row, lim)
    else:
        _resize(row, lim)


@njit(cache=True, nogil=True, fastmath=True)
def _change_color(row, lim):  # type: ignore[no-untyped-def]
    for channel in range(RED, BLUE + 1):
        row[channel] = clamp(row[channel] + np.random.normal(0.0, 0.06), 0.0, 1.0)
    row[ALPHA] = clamp(row[ALPHA] + np.random.normal(0.0, 0.06), lim[L_MIN_ALPHA], lim[L_MAX_ALPHA])


@njit(cache=True, nogil=True, fastmath=True)
def _random_row_seeded(seed, lim, cx, cy, r, g, b):  # type: ignore[no-untyped-def]
    np.random.seed(seed)
    return _random_row(lim, cx, cy, r, g, b)


@njit(cache=True, nogil=True)
def _mutate(genome, seed, rate, per_row, lim):  # type: ignore[no-untyped-def]
    """Edit the shape of some elements, then the colour of some, add one, remove one, move one. ``seed`` starts the
    thread's Numba generator, so a call depends only on its arguments. ``genome`` is not modified."""
    np.random.seed(seed)
    g = genome.copy()
    picked = np.empty(g.shape[0], dtype=np.int64)
    for t in range(pick(g.shape[0], per_row, picked)):
        _edit_shape(g[picked[t]], lim)
    for t in range(pick(g.shape[0], per_row, picked)):
        _change_color(g[picked[t]], lim)
    if np.random.random() < rate and g.shape[0] < lim[L_MAX_ELEMENTS]:
        g = insert_row(g, np.random.randint(0, g.shape[0] + 1), _new_row(lim))
    return drop_and_move(g, rate, lim, True)


def random_row(rng: Random, limits: Limits, rgb: tuple[float, float, float] | None = None,
               center: tuple[float, float] | None = None) -> Row:
    """Random slightly curved stroke centred on ``center`` (random if omitted), at most ``max_size`` long."""
    cx, cy = center if center is not None else (-1.0, -1.0)  # -1: draw it at random
    r, g, b = rgb if rgb is not None else (-1.0, -1.0, -1.0)
    return _random_row_seeded(rng.getrandbits(32), limits_array(limits), cx, cy, r, g, b)


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_element: float | None = None) -> Genome:
    """Bend, move or thicken some strokes, recolour some, add / remove / reorder (see ``kernels.mutate_rows``)."""
    return _mutate(np.ascontiguousarray(genome, dtype=np.float32), rng.getrandbits(32), rate,
                   rate if per_element is None else per_element, limits_array(limits))


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
    seg_x = np.empty(_MAX_SEGMENTS, dtype=np.float64)
    seg_y = np.empty(_MAX_SEGMENTS, dtype=np.float64)
    seg_vx = np.empty(_MAX_SEGMENTS, dtype=np.float64)
    seg_vy = np.empty(_MAX_SEGMENTS, dtype=np.float64)
    seg_inv = np.empty(_MAX_SEGMENTS, dtype=np.float64)
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
        for i in range(n):  # segment i as its start, its vector and 1 / its squared length (0 for a point)
            seg_x[i], seg_y[i] = pts_x[i], pts_y[i]
            seg_vx[i], seg_vy[i] = pts_x[i + 1] - pts_x[i], pts_y[i + 1] - pts_y[i]
            span = seg_vx[i] * seg_vx[i] + seg_vy[i] * seg_vy[i]
            seg_inv[i] = 0.0 if span == 0.0 else 1.0 / span
        for py in range(y0, y1):
            yc = py + 0.5
            for px in range(x0, x1):
                xc = px + 0.5
                for i in range(n):  # the pixel is painted if its centre is within the brush of any segment
                    t = min(1.0, max(0.0, ((xc - seg_x[i]) * seg_vx[i] + (yc - seg_y[i]) * seg_vy[i]) * seg_inv[i]))
                    dx, dy = xc - (seg_x[i] + t * seg_vx[i]), yc - (seg_y[i] + t * seg_vy[i])
                    if dx * dx + dy * dy <= limit:
                        canvas[py - oy, px - ox, 0] = canvas[py - oy, px - ox, 0] * keep + r
                        canvas[py - oy, px - ox, 1] = canvas[py - oy, px - ox, 1] * keep + g
                        canvas[py - oy, px - ox, 2] = canvas[py - oy, px - ox, 2] * keep + b
                        break


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
    pts_x = np.empty(_MAX_SEGMENTS + 1, dtype=np.float64)
    pts_y = np.empty(_MAX_SEGMENTS + 1, dtype=np.float64)
    areas = np.empty((n_old + n_new, 4), dtype=np.int64)
    for k in range(n_old + n_new):
        at = k
        row = before[old_rows[k]] if k < n_old else after[new_rows[k - n_old]]
        _, _, areas[at, 0], areas[at, 1], areas[at, 2], areas[at, 3] = _curve(row, width, height, pts_x, pts_y)
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
