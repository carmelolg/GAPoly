"""Numba helpers shared by the rasterisers of every shape. The kernels release the GIL (``nogil``), so threads
evaluate several genomes in parallel."""
from __future__ import annotations

import numpy as np
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


@njit(cache=True, nogil=True, fastmath=True)
def window_error(canvas, target, x0, y0):  # type: ignore[no-untyped-def]
    """Sum of squared errors between ``canvas`` and the window of ``target`` whose top left corner is ``(x0, y0)``.

    Same arithmetic as ``canvas_mse`` (squares in float32, sum in float64), so a window as big as the image gives
    exactly the sum ``canvas_mse`` divides by the number of values."""
    total = 0.0
    for y in range(canvas.shape[0]):
        for x in range(canvas.shape[1]):
            for c in range(3):
                d = canvas[y, x, c] - target[y0 + y, x0 + x, c]
                total += d * d
    return total


@njit(cache=True, nogil=True)
def _same_row(a, b):  # type: ignore[no-untyped-def]
    for c in range(a.shape[0]):
        if a[c] != b[c]:
            return False
    return True


@njit(cache=True, nogil=True)
def unmatched_rows(parent, child, limit):  # type: ignore[no-untyped-def]
    """Rows of ``parent`` and of ``child`` that cannot be paired with an identical row of the other genome.

    The pairing keeps the order (a greedy walk that looks one row ahead on each side, so an inserted, a removed,
    an edited or a moved element leaves only itself unpaired). Whatever pairing is used is correct for the error:
    paired rows are identical and in the same order, so pixels no unpaired row reaches are painted the same way
    in both genomes. A tighter pairing only means a smaller area to redraw.

    Returns ``(parent_rows, child_rows, n_parent, n_child)``: the first ``n_*`` entries of each array are row
    indices. ``n_parent`` is -1 when more than ``limit`` rows are unpaired on either side (not worth it)."""
    parent_rows = np.empty(limit, dtype=np.int64)
    child_rows = np.empty(limit, dtype=np.int64)
    n_parent = 0
    n_child = 0
    i = 0
    j = 0
    while i < parent.shape[0] or j < child.shape[0]:
        take_parent = False
        take_child = False
        if i >= parent.shape[0]:
            take_child = True
        elif j >= child.shape[0]:
            take_parent = True
        elif _same_row(parent[i], child[j]):
            i += 1
            j += 1
            continue
        elif j + 1 < child.shape[0] and _same_row(parent[i], child[j + 1]):
            take_child = True  # a row was inserted
        elif i + 1 < parent.shape[0] and _same_row(parent[i + 1], child[j]):
            take_parent = True  # a row was removed
        else:
            take_parent = True  # a row was edited: both sides are unpaired
            take_child = True
        if take_parent:
            if n_parent == limit:
                return parent_rows, child_rows, -1, -1
            parent_rows[n_parent] = i
            n_parent += 1
            i += 1
        if take_child:
            if n_child == limit:
                return parent_rows, child_rows, -1, -1
            child_rows[n_child] = j
            n_child += 1
            j += 1
    return parent_rows, child_rows, n_parent, n_child


@njit(cache=True, nogil=True)
def merge_boxes(boxes):  # type: ignore[no-untyped-def]
    """Replace boxes ``(x0, x1, y0, y1)`` that overlap by their bounding box, until none overlap. Empty boxes
    (``x1 <= x0`` or ``y1 <= y0``) are dropped. Returns the surviving boxes, a new ``(m, 4)`` int64 array."""
    count = boxes.shape[0]
    work = np.empty((count, 4), dtype=np.int64)
    alive = np.zeros(count, dtype=np.bool_)
    for k in range(count):
        work[k] = boxes[k]
        alive[k] = boxes[k, 1] > boxes[k, 0] and boxes[k, 3] > boxes[k, 2]
    merged = True
    while merged:
        merged = False
        for a in range(count):
            if not alive[a]:
                continue
            for b in range(a + 1, count):
                if not alive[b]:
                    continue
                if work[a, 0] < work[b, 1] and work[b, 0] < work[a, 1] and work[a, 2] < work[b, 3] and work[b, 2] < work[a, 3]:
                    work[a, 0] = min(work[a, 0], work[b, 0])
                    work[a, 1] = max(work[a, 1], work[b, 1])
                    work[a, 2] = min(work[a, 2], work[b, 2])
                    work[a, 3] = max(work[a, 3], work[b, 3])
                    alive[b] = False
                    merged = True
    result = np.empty((int(alive.sum()), 4), dtype=np.int64)
    m = 0
    for k in range(count):
        if alive[k]:
            result[m] = work[k]
            m += 1
    return result
