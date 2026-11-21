"""Polygons: genome layout, rasteriser and operators.

A genome is a float32 array of shape ``(polygons, COLS)``: one row per polygon, drawn top to bottom in row
order (later rows on top). Every value is in [0, 1]; coordinates are fractions of the image side, so a genome
renders at any resolution.

Layout of one row (17 columns)::

    column:   0 | 1   2 | 3   4 | ...  | 11  12 | 13  14  15  16
    meaning:  n | x0  y0  x1  y1  ...    x5  y5 |  r   g   b   a

- ``n`` is how many vertices are used (3 to 6); the vertices after the n-th are ignored.
- ``r g b`` is the colour, ``a`` the opacity of the polygon (0 transparent, 1 opaque).

Example: a half-transparent red triangle::

    [3,  0.10 0.10,  0.30 0.10,  0.20 0.30,  0 0 0 0 0 0,  1.0 0.0 0.0 0.5]

Never index a row with bare numbers: use ``N_VERTICES``, ``X(v)``, ``Y(v)``, ``RED``..``ALPHA``, ``vertices``,
``color`` and ``make_row``. ``describe(genome)`` prints a genome as text, handy when debugging.

The rasteriser samples pixels at their centres and a polygon covers a pixel by the even-odd rule.
"""
from __future__ import annotations

from random import Random

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..core.common import Array, Genome, Limits, Row
from ..core.kernels import (
    L_MAX_ALPHA,
    L_MAX_ELEMENTS,
    L_MAX_SIZE,
    L_MAX_VERTICES,
    L_MIN_ALPHA,
    L_MIN_VERTICES,
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

MAX_VERTICES = 6

N_VERTICES = 0  # column holding the number of vertices used
VERTEX_START = 1  # first vertex coordinate; vertex v is at columns X(v), Y(v)
RED = VERTEX_START + 2 * MAX_VERTICES  # 13
GREEN, BLUE, ALPHA = RED + 1, RED + 2, RED + 3
COLS = ALPHA + 1  # 17


def X(vertex: int) -> int:
    """Column of the x coordinate of vertex number ``vertex`` (0-based)."""
    return VERTEX_START + 2 * vertex


def Y(vertex: int) -> int:
    """Column of the y coordinate of vertex number ``vertex`` (0-based)."""
    return VERTEX_START + 2 * vertex + 1


def make_row(points: list[tuple[float, float]], rgba: tuple[float, float, float, float]) -> Row:
    """Build one polygon row from its vertices ``[(x, y), ...]`` and its colour ``(r, g, b, a)``."""
    row = np.zeros(COLS, dtype=np.float32)
    row[N_VERTICES] = len(points)
    for v, (x, y) in enumerate(points):
        row[X(v)], row[Y(v)] = x, y
    row[RED : ALPHA + 1] = rgba
    return row


def vertices(row: Row) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The used x and y coordinates of a polygon (views into the row)."""
    n = int(row[N_VERTICES])
    return row[VERTEX_START : VERTEX_START + 2 * n : 2], row[VERTEX_START + 1 : VERTEX_START + 2 * n : 2]


def color(row: Row) -> NDArray[np.float32]:
    """The ``(r, g, b, a)`` of a polygon (a view into the row)."""
    return row[RED : ALPHA + 1]


def extent(row: Row) -> float:
    xs, ys = vertices(row)
    return float(max(xs.max() - xs.min(), ys.max() - ys.min()))


def describe(genome: Genome) -> str:
    """One line per polygon, in drawing order: vertices and colour."""
    lines = []
    for k, row in enumerate(genome):
        xs, ys = vertices(row)
        points = " ".join(f"({x:.2f},{y:.2f})" for x, y in zip(xs, ys, strict=True))
        r, g, b, a = color(row)
        lines.append(f"#{k} {len(xs)} vertices {points} rgba({r:.2f},{g:.2f},{b:.2f},{a:.2f})")
    return "\n".join(lines)


@njit(cache=True, nogil=True, fastmath=True)
def _extent(row):  # type: ignore[no-untyped-def]
    n = int(row[N_VERTICES])
    min_x = max_x = row[VERTEX_START]
    min_y = max_y = row[VERTEX_START + 1]
    for v in range(1, n):
        x, y = row[VERTEX_START + 2 * v], row[VERTEX_START + 2 * v + 1]
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True, nogil=True, fastmath=True)
def _random_row(lim, cx, cy, r, g, b):  # type: ignore[no-untyped-def]
    """Small random polygon around ``(cx, cy)`` with colour ``(r, g, b)``; a negative value is drawn at random."""
    spread = lim[L_MAX_SIZE] / 2
    if cx < 0.0:
        cx, cy = np.random.random(), np.random.random()
    if r < 0.0:
        r, g, b = np.random.random(), np.random.random(), np.random.random()
    n = np.random.randint(int(lim[L_MIN_VERTICES]), int(lim[L_MAX_VERTICES]) + 1)
    row = np.zeros(COLS, dtype=np.float32)
    row[N_VERTICES] = n
    for v in range(n):
        row[VERTEX_START + 2 * v] = clamp(cx + np.random.uniform(-spread, spread), 0.0, 1.0)
        row[VERTEX_START + 2 * v + 1] = clamp(cy + np.random.uniform(-spread, spread), 0.0, 1.0)
    row[RED], row[GREEN], row[BLUE] = r, g, b
    row[ALPHA] = np.random.uniform(max(lim[L_MIN_ALPHA], 0.1), min(lim[L_MAX_ALPHA], 0.6))
    return row


@njit(cache=True, nogil=True, fastmath=True)
def _new_row(lim):  # type: ignore[no-untyped-def]
    return _random_row(lim, -1.0, -1.0, -1.0, -1.0, -1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _move_vertex(row, lim):  # type: ignore[no-untyped-def]
    """Move one vertex in place; undone if the polygon would grow past the cap."""
    before = row.copy()
    old = _extent(row)
    v = np.random.randint(0, int(row[N_VERTICES]))
    row[VERTEX_START + 2 * v] = clamp(row[VERTEX_START + 2 * v] + np.random.normal(0.0, 0.015), 0.0, 1.0)
    row[VERTEX_START + 2 * v + 1] = clamp(row[VERTEX_START + 2 * v + 1] + np.random.normal(0.0, 0.015), 0.0, 1.0)
    if _extent(row) > max(lim[L_MAX_SIZE], old):  # never grow past the cap
        row[:] = before


@njit(cache=True, nogil=True, fastmath=True)
def _shift_polygon(row, lim):  # type: ignore[no-untyped-def]
    dx, dy = np.random.normal(0.0, 0.02), np.random.normal(0.0, 0.02)
    for v in range(int(row[N_VERTICES])):
        row[VERTEX_START + 2 * v] = clamp(row[VERTEX_START + 2 * v] + dx, 0.0, 1.0)
        row[VERTEX_START + 2 * v + 1] = clamp(row[VERTEX_START + 2 * v + 1] + dy, 0.0, 1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _edit_shape(row, lim):  # type: ignore[no-untyped-def]
    if np.random.random() < 0.7:
        _move_vertex(row, lim)
    else:
        _shift_polygon(row, lim)


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
    """Small random polygon around ``center`` (random if omitted). ``rgb`` is the colour; alpha is random."""
    cx, cy = center if center is not None else (-1.0, -1.0)  # -1: draw it at random
    r, g, b = rgb if rgb is not None else (-1.0, -1.0, -1.0)
    return _random_row_seeded(rng.getrandbits(32), limits_array(limits), cx, cy, r, g, b)


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_element: float | None = None) -> Genome:
    """Move or reshape some polygons, recolour some, add / remove / reorder (see ``kernels.mutate_rows``)."""
    return _mutate(np.ascontiguousarray(genome, dtype=np.float32), rng.getrandbits(32), rate,
                   rate if per_element is None else per_element, limits_array(limits))


@njit(cache=True, nogil=True, fastmath=True)
def _box(row, width, height):  # type: ignore[no-untyped-def]
    """Pixels ``(x0, x1, y0, y1)`` a polygon row can paint (empty for fewer than 3 vertices)."""
    n = int(row[N_VERTICES])
    if n < 3:
        return 0, 0, 0, 0
    min_x, max_x = width, 0.0
    min_y, max_y = height, 0.0
    for v in range(n):
        x = row[VERTEX_START + 2 * v] * width
        y = row[VERTEX_START + 2 * v + 1] * height
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
    return (max(0, int(np.floor(min_x))), min(width, int(np.ceil(max_x)) + 1),
            max(0, int(np.floor(min_y))), min(height, int(np.ceil(max_y)) + 1))


@njit(cache=True, nogil=True, fastmath=True)
def _draw(genome, canvas, ox, oy, width, height):  # type: ignore[no-untyped-def]
    """Paint the polygons on ``canvas``, a window of a ``width`` x ``height`` image with its corner at ``(ox, oy)``."""
    xs = np.empty(MAX_VERTICES, dtype=np.float32)
    ys = np.empty(MAX_VERTICES, dtype=np.float32)
    for k in range(genome.shape[0]):
        n = int(genome[k, N_VERTICES])
        if n < 3:
            continue
        x0, x1, y0, y1 = _box(genome[k], width, height)
        x0, x1 = max(x0, ox), min(x1, ox + canvas.shape[1])
        y0, y1 = max(y0, oy), min(y1, oy + canvas.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        for v in range(n):
            xs[v] = genome[k, VERTEX_START + 2 * v] * width
            ys[v] = genome[k, VERTEX_START + 2 * v + 1] * height
        alpha = genome[k, ALPHA]
        keep = 1.0 - alpha
        r = genome[k, RED] * alpha
        g = genome[k, GREEN] * alpha
        b = genome[k, BLUE] * alpha
        for py in range(y0, y1):
            yc = py + 0.5
            for px in range(x0, x1):
                xc = px + 0.5
                inside = False
                j = n - 1
                for i in range(n):
                    if (ys[i] > yc) != (ys[j] > yc) and xc < (xs[j] - xs[i]) * (yc - ys[i]) / (ys[j] - ys[i]) + xs[i]:
                        inside = not inside
                    j = i
                if inside:
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


class PolygonShape(Shape):
    name = "polygon"
    plural = "polygons"
    cols = COLS
    uses_size = True
    default_elements = 400
    default_initial = 60

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
