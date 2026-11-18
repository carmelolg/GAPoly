"""Polygon genome layout.

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
"""
from __future__ import annotations

from random import Random

import numpy as np
from numpy.typing import NDArray

from .limits import Limits, clamp

MAX_VERTICES = 6

N_VERTICES = 0  # column holding the number of vertices used
VERTEX_START = 1  # first vertex coordinate; vertex v is at columns X(v), Y(v)
RED = VERTEX_START + 2 * MAX_VERTICES  # 13
GREEN, BLUE, ALPHA = RED + 1, RED + 2, RED + 3
COLS = ALPHA + 1  # 17

Genome = NDArray[np.float32]
Row = NDArray[np.float32]


def X(vertex: int) -> int:
    """Column of the x coordinate of vertex number ``vertex`` (0-based)."""
    return VERTEX_START + 2 * vertex


def Y(vertex: int) -> int:
    """Column of the y coordinate of vertex number ``vertex`` (0-based)."""
    return VERTEX_START + 2 * vertex + 1


def empty_genome() -> Genome:
    return np.zeros((0, COLS), dtype=np.float32)


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
