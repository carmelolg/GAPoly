"""Numba helpers shared by the rasterisers of every shape. The kernels release the GIL (``nogil``), so threads
evaluate several genomes in parallel."""
from __future__ import annotations

from numba import njit


@njit(cache=True, nogil=True, fastmath=True)
def fill(canvas, background):  # type: ignore[no-untyped-def]
    """Paint the whole canvas with the background colour."""
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
