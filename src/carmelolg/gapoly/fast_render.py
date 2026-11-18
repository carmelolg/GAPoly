"""Numba rasteriser for polygons: draws a whole genome in one compiled loop (no Python per polygon, no allocations).

Pixels are sampled at their centres and a polygon covers a pixel by the even-odd rule. The kernels release the
GIL (``nogil``), so threads evaluate several genomes in parallel.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .shapes import (  # column layout: see shapes.py
    ALPHA,
    BLUE,
    GREEN,
    MAX_VERTICES,
    N_VERTICES,
    RED,
    VERTEX_START,
)


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
def fill(canvas, background):  # type: ignore[no-untyped-def]
    """Paint the whole canvas with the background colour (shared by every kind of element)."""
    for y in range(canvas.shape[0]):
        for x in range(canvas.shape[1]):
            for c in range(3):
                canvas[y, x, c] = background[c]


@njit(cache=True, nogil=True, fastmath=True)
def canvas_mse(canvas, target):  # type: ignore[no-untyped-def]
    """Mean squared error between two float32 (h, w, 3) images."""
    total = 0.0
    for y in range(target.shape[0]):
        for x in range(target.shape[1]):
            for c in range(3):
                d = canvas[y, x, c] - target[y, x, c]
                total += d * d
    return total / target.size


@njit(cache=True, nogil=True, fastmath=True)
def render_into(genome, background, canvas):  # type: ignore[no-untyped-def]
    fill(canvas, background)
    _draw(genome, canvas)


@njit(cache=True, nogil=True, fastmath=True)
def render_mse(genome, target, background):  # type: ignore[no-untyped-def]
    """Mean squared error between the rendered genome and ``target`` (float32, (h, w, 3))."""
    canvas = np.empty_like(target)
    render_into(genome, background, canvas)
    return canvas_mse(canvas, target)


def render_image(genome: np.ndarray, width: int, height: int, background: np.ndarray) -> np.ndarray:
    canvas = np.empty((height, width, 3), dtype=np.float32)
    render_into(np.ascontiguousarray(genome, dtype=np.float32), background, canvas)
    return canvas
