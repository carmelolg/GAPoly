"""Voronoi points: genome layout, rasteriser and operators.

A genome is a float32 array of shape ``(points, COLS)``, one row per point (a "seed"). Every pixel takes the colour
of the nearest seed, so the image is a mosaic of flat-coloured cells. Every value is in [0, 1]; ``x`` and ``y`` are
fractions of the image width and height, so a genome renders at any resolution.

Layout of one row (5 columns)::

    column:   0   1 | 2  3  4
    meaning:  x   y | r  g  b

There is no alpha and no size, and the order of the rows does not matter. With no seed the image is the background.

The rasteriser puts the seeds in a grid and looks at a pixel's own cell and its neighbours, ring by ring, until no
closer seed can exist: the result is exact and costs roughly the number of pixels, not pixels times seeds.
"""
from __future__ import annotations

from random import Random

import numpy as np
from numba import njit

from .common import Array, Genome, Limits, Row, clamp
from .kernels import canvas_mse, fill
from .operators import mutate_rows
from .shape import Shape

X, Y = 0, 1
RED, GREEN, BLUE = 2, 3, 4
COLS = 5


def make_row(x: float, y: float, rgb: tuple[float, float, float]) -> Row:
    """One seed row from its position and colour ``(r, g, b)``."""
    return np.array([x, y, *rgb], dtype=np.float32)


def describe(genome: Genome) -> str:
    """One line per seed."""
    return "\n".join(f"#{k} seed ({row[X]:.2f},{row[Y]:.2f}) rgb({row[RED]:.2f},{row[GREEN]:.2f},{row[BLUE]:.2f})"
                     for k, row in enumerate(genome))


def random_row(rng: Random, limits: Limits, rgb: tuple[float, float, float] | None = None,
               center: tuple[float, float] | None = None) -> Row:
    """Seed at ``center`` (random if omitted) with colour ``rgb`` (random if omitted)."""
    cx, cy = center if center is not None else (rng.random(), rng.random())
    rgb = rgb if rgb is not None else (rng.random(), rng.random(), rng.random())
    return make_row(cx, cy, rgb)


def move(row: np.ndarray, rng: Random, limits: Limits) -> None:
    """Mostly a small nudge, sometimes a jump to a different spot."""
    sigma = 0.01 if rng.random() < 0.7 else 0.05
    row[X] = clamp(float(row[X]) + rng.gauss(0, sigma))
    row[Y] = clamp(float(row[Y]) + rng.gauss(0, sigma))


def change_color(row: np.ndarray, rng: Random, limits: Limits, sigma: float = 0.06) -> None:
    for channel in (RED, GREEN, BLUE):
        row[channel] = clamp(float(row[channel]) + rng.gauss(0, sigma))


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_point: float | None = None) -> Genome:
    """Move some seeds, recolour some, add / remove (the order does not matter, so there is no reorder)."""
    return mutate_rows(genome, rng, rate, limits, per_point, move, change_color,
                       lambda r, lim: random_row(r, lim), reorder=False)


@njit(cache=True, nogil=True, fastmath=True)
def _draw(genome, canvas):  # type: ignore[no-untyped-def]
    height, width = canvas.shape[0], canvas.shape[1]
    n = genome.shape[0]
    if n == 0:
        return
    cell = max(1.0, np.sqrt(width * height * 2.0 / n))  # about two seeds per grid cell
    gw = int(np.ceil(width / cell))
    gh = int(np.ceil(height / cell))
    sx = np.empty(n, dtype=np.float64)
    sy = np.empty(n, dtype=np.float64)
    where = np.empty(n, dtype=np.int64)
    start = np.zeros(gw * gh + 1, dtype=np.int64)
    for k in range(n):
        sx[k] = genome[k, X] * width
        sy[k] = genome[k, Y] * height
        where[k] = min(gh - 1, int(sy[k] / cell)) * gw + min(gw - 1, int(sx[k] / cell))
        start[where[k] + 1] += 1
    for c in range(gw * gh):
        start[c + 1] += start[c]
    order = np.empty(n, dtype=np.int64)
    next_slot = start[:-1].copy()
    for k in range(n):
        order[next_slot[where[k]]] = k
        next_slot[where[k]] += 1
    last_ring = max(gw, gh)
    for py in range(height):
        yc = py + 0.5
        cy = min(gh - 1, int(yc / cell))
        for px in range(width):
            xc = px + 0.5
            cx = min(gw - 1, int(xc / cell))
            best = 1e300
            nearest = 0
            ring = 0
            while ring <= last_ring:
                for gy in range(cy - ring, cy + ring + 1):
                    if gy < 0 or gy >= gh:
                        continue
                    whole_row = gy == cy - ring or gy == cy + ring
                    step = 1 if whole_row or ring == 0 else 2 * ring  # inner rows: only the two ends of the ring
                    for gx in range(cx - ring, cx + ring + 1, step):
                        if gx < 0 or gx >= gw:
                            continue
                        c = gy * gw + gx
                        for i in range(start[c], start[c + 1]):
                            k = order[i]
                            d = (sx[k] - xc) ** 2 + (sy[k] - yc) ** 2
                            if d < best:
                                best = d
                                nearest = k
                if best <= (ring * cell) ** 2:  # a seed in a farther ring is at least ring * cell away
                    break
                ring += 1
            canvas[py, px, 0] = genome[nearest, RED]
            canvas[py, px, 1] = genome[nearest, GREEN]
            canvas[py, px, 2] = genome[nearest, BLUE]


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


class VoronoiShape(Shape):
    name = "voronoi"
    plural = "seeds"
    cols = COLS
    uses_size = False
    default_elements = 1000
    default_initial = 150

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
