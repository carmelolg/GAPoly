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

from ..core.common import Array, Genome, Limits, Row, clamp
from ..core.kernels import canvas_mse, fill
from ..core.operators import mutate_rows
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


def random_row(rng: Random, limits: Limits, rgb: tuple[float, float, float] | None = None,
               center: tuple[float, float] | None = None) -> Row:
    """Small random polygon around ``center`` (random if omitted). ``rgb`` is the colour; alpha is random."""
    spread = limits.max_size / 2
    cx, cy = center if center is not None else (rng.random(), rng.random())
    n = rng.randint(limits.min_vertices, limits.max_vertices)
    points = [(clamp(cx + rng.uniform(-spread, spread)), clamp(cy + rng.uniform(-spread, spread))) for _ in range(n)]
    rgb = rgb if rgb is not None else (rng.random(), rng.random(), rng.random())
    return make_row(points, (*rgb, rng.uniform(max(limits.min_alpha, 0.1), min(limits.max_alpha, 0.6))))


def move_vertex(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.015) -> None:
    """Move one vertex in place (``row`` is a private copy); undone if the polygon would grow past the cap."""
    before, old = row.copy(), extent(row)
    v = rng.randrange(int(row[N_VERTICES]))
    row[X(v)] = clamp(float(row[X(v)]) + rng.gauss(0, sigma))
    row[Y(v)] = clamp(float(row[Y(v)]) + rng.gauss(0, sigma))
    if extent(row) > max(limits.max_size, old):  # never grow past the cap
        row[:] = before


def shift_polygon(row: np.ndarray, rng: Random, sigma: float = 0.02) -> None:
    dx, dy = rng.gauss(0, sigma), rng.gauss(0, sigma)
    for v in range(int(row[N_VERTICES])):
        row[X(v)] = clamp(float(row[X(v)]) + dx)
        row[Y(v)] = clamp(float(row[Y(v)]) + dy)


def change_color(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.06) -> None:
    for channel in range(RED, GREEN + 2):  # red, green, blue
        row[channel] = clamp(float(row[channel]) + rng.gauss(0, sigma))
    row[ALPHA] = clamp(float(row[ALPHA]) + rng.gauss(0, sigma), limits.min_alpha, limits.max_alpha)


def _edit_polygon(row: np.ndarray, rng: Random, limits: Limits) -> None:
    if rng.random() < 0.7:
        move_vertex(row, rng, limits)
    else:
        shift_polygon(row, rng)


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_polygon: float | None = None) -> Genome:
    """Move or reshape some polygons, recolour some, add / remove / reorder (see ``operators.mutate_rows``)."""
    return mutate_rows(genome, rng, rate, limits, per_polygon, _edit_polygon, change_color,
                       lambda r, lim: random_row(r, lim))


@njit(cache=True, nogil=True, fastmath=True)
def _draw(genome, canvas):  # type: ignore[no-untyped-def]
    height, width = canvas.shape[0], canvas.shape[1]
    xs = np.empty(MAX_VERTICES, dtype=np.float32)
    ys = np.empty(MAX_VERTICES, dtype=np.float32)
    for k in range(genome.shape[0]):
        n = int(genome[k, N_VERTICES])
        if n < 3:
            continue
        min_x, max_x = width, 0.0
        min_y, max_y = height, 0.0
        for v in range(n):
            xs[v] = genome[k, VERTEX_START + 2 * v] * width
            ys[v] = genome[k, VERTEX_START + 2 * v + 1] * height
            min_x, max_x = min(min_x, xs[v]), max(max_x, xs[v])
            min_y, max_y = min(min_y, ys[v]), max(max_y, ys[v])
        x0, x1 = max(0, int(np.floor(min_x))), min(width, int(np.ceil(max_x)) + 1)
        y0, y1 = max(0, int(np.floor(min_y))), min(height, int(np.ceil(max_y)) + 1)
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


class PolygonShape(Shape):
    name = "polygon"
    plural = "polygons"
    cols = COLS
    uses_size = True
    default_elements = 400
    default_initial = 60

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
