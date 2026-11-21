"""Glyphs: typographic characters, genome layout, rasteriser and operators.

A genome is a float32 array of shape ``(glyphs, COLS)``, one row per character, drawn in row order (later rows on
top). Every value is in [0, 1]; ``x`` and ``y`` are fractions of the image width and height, ``size`` is the side of the
character's square box as a fraction of the longest image side, so a genome renders at any resolution.

Layout of one row (9 columns)::

    column:   0   1 |  2   |   3   |   4   | 5  6  7 |  8
    meaning:  x   y | size | angle | glyph | r  g  b | alpha

- ``angle`` is the rotation as a fraction of a full turn (0 upright, 0.25 a quarter turn clockwise).
- ``glyph`` picks the character: ``GLYPHS[int(glyph * len(GLYPHS))]``. Mutation draws a new character at random.
- ``r g b`` is the ink colour and ``alpha`` its opacity.

The characters are rasterised once from a font into an atlas of coverage masks (``atlas()``); the Numba kernel samples
it bilinearly for every pixel of a glyph's box, so a glyph is drawn at any size and angle. The shape of the characters
depends on the font Pillow finds, so tests must not depend on the exact pixels of a character.
"""
from __future__ import annotations

from functools import cache
from random import Random

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont

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

X, Y, SIZE, ANGLE, GLYPH = 0, 1, 2, 3, 4
RED, GREEN, BLUE, ALPHA = 5, 6, 7, 8
COLS = 9

GLYPHS = "#%&*+=?@/\\|<>()[]{}~^:;-_oOxXzZ01"  # dark and light characters, round and angular ones
ATLAS_SIZE = 48  # side in pixels of each character mask

_MIN_HALF_PX = 0.7072  # a glyph box always reaches the centre of the pixel it sits in


def make_row(x: float, y: float, size: float, angle: float, glyph: float,
             rgba: tuple[float, float, float, float]) -> Row:
    """One glyph row from its centre, size, rotation, character (0-1) and colour ``(r, g, b, a)``."""
    return np.array([x, y, size, angle, glyph, *rgba], dtype=np.float32)


def glyph_char(value: float) -> str:
    """The character a ``glyph`` column value stands for."""
    return GLYPHS[min(int(value * len(GLYPHS)), len(GLYPHS) - 1)]


def describe(genome: Genome) -> str:
    """One line per glyph, in drawing order."""
    return "\n".join(
        f"#{k} glyph {glyph_char(float(row[GLYPH]))!r} ({row[X]:.2f},{row[Y]:.2f}) size {row[SIZE]:.3f} "
        f"angle {row[ANGLE]:.2f} rgba({row[RED]:.2f},{row[GREEN]:.2f},{row[BLUE]:.2f},{row[ALPHA]:.2f})"
        for k, row in enumerate(genome))


def _font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=int(ATLAS_SIZE * 0.85))  # scalable font (Pillow 10.1+ with FreeType)
    except (TypeError, OSError):  # pragma: no cover - old Pillow or no FreeType: the small bitmap font
        return ImageFont.load_default()


@cache
def atlas() -> Array:
    """Coverage masks of every character in ``GLYPHS``: float32 array ``(len(GLYPHS), ATLAS_SIZE, ATLAS_SIZE)``."""
    font = _font()
    masks = np.zeros((len(GLYPHS), ATLAS_SIZE, ATLAS_SIZE), dtype=np.float32)
    for i, char in enumerate(GLYPHS):
        image = Image.new("L", (ATLAS_SIZE, ATLAS_SIZE), 0)
        draw = ImageDraw.Draw(image)
        try:
            draw.text((ATLAS_SIZE / 2, ATLAS_SIZE / 2), char, font=font, fill=255, anchor="mm")
        except ValueError:  # pragma: no cover - bitmap font: no anchors, centre by hand
            left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
            draw.text(((ATLAS_SIZE - right - left) / 2, (ATLAS_SIZE - bottom - top) / 2), char, font=font, fill=255)
        masks[i] = np.asarray(image, dtype=np.float32) / 255.0
    return masks


@njit(cache=True, nogil=True, fastmath=True)
def _size_bounds(lim):  # type: ignore[no-untyped-def]
    low = max(lim[L_MIN_SIZE], 1e-4)
    return low, max(low, lim[L_MAX_SIZE])


@njit(cache=True, nogil=True, fastmath=True)
def _random_row(lim, cx, cy, r, g, b):  # type: ignore[no-untyped-def]
    """Random upright glyph at ``(cx, cy)``, at most ``max_size`` wide; negative values are drawn at random."""
    low, high = _size_bounds(lim)
    if cx < 0.0:
        cx, cy = np.random.random(), np.random.random()
    if r < 0.0:
        r, g, b = np.random.random(), np.random.random(), np.random.random()
    row = np.empty(COLS, dtype=np.float32)
    row[X], row[Y] = cx, cy
    row[SIZE] = np.random.uniform(low, high)
    row[ANGLE] = 0.0
    row[GLYPH] = np.random.random()
    row[RED], row[GREEN], row[BLUE] = r, g, b
    row[ALPHA] = np.random.uniform(max(lim[L_MIN_ALPHA], 0.1), min(lim[L_MAX_ALPHA], 0.6))
    return row


@njit(cache=True, nogil=True, fastmath=True)
def _new_row(lim):  # type: ignore[no-untyped-def]
    return _random_row(lim, -1.0, -1.0, -1.0, -1.0, -1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _move(row, lim):  # type: ignore[no-untyped-def]
    row[X] = clamp(row[X] + np.random.normal(0.0, 0.015), 0.0, 1.0)
    row[Y] = clamp(row[Y] + np.random.normal(0.0, 0.015), 0.0, 1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _resize(row, lim):  # type: ignore[no-untyped-def]
    """Scale the size by a log-normal factor; never grows past the cap (a glyph already above it can only shrink)."""
    low, high = _size_bounds(lim)
    row[SIZE] = clamp(row[SIZE] * np.exp(np.random.normal(0.0, 0.25)), low, max(high, row[SIZE]))


@njit(cache=True, nogil=True, fastmath=True)
def _rotate(row, lim):  # type: ignore[no-untyped-def]
    row[ANGLE] = (row[ANGLE] + np.random.normal(0.0, 0.03)) % 1.0  # a turn wraps around


@njit(cache=True, nogil=True, fastmath=True)
def _edit_shape(row, lim):  # type: ignore[no-untyped-def]
    pick = np.random.random()
    if pick < 0.35:
        _move(row, lim)
    elif pick < 0.65:
        _resize(row, lim)
    elif pick < 0.85:
        _rotate(row, lim)
    else:
        row[GLYPH] = np.random.random()


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
    """Random upright glyph at ``center`` (random if omitted), at most ``max_size`` wide. ``rgb`` is the ink."""
    cx, cy = center if center is not None else (-1.0, -1.0)  # -1: draw it at random
    r, g, b = rgb if rgb is not None else (-1.0, -1.0, -1.0)
    return _random_row_seeded(rng.getrandbits(32), limits_array(limits), cx, cy, r, g, b)


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_element: float | None = None) -> Genome:
    """Move, resize, rotate or change the character of some glyphs, recolour some, add / remove / reorder."""
    return _mutate(np.ascontiguousarray(genome, dtype=np.float32), rng.getrandbits(32), rate,
                   rate if per_element is None else per_element, limits_array(limits))


@njit(cache=True, nogil=True, fastmath=True)
def _sample(mask, u, v):  # type: ignore[no-untyped-def]
    """Bilinear coverage of one character mask at ``(u, v)`` in [0, 1) box coordinates (outside = 0)."""
    n = mask.shape[0]
    fx = u * n - 0.5
    fy = v * n - 0.5
    ix = int(np.floor(fx))
    iy = int(np.floor(fy))
    tx = fx - ix
    ty = fy - iy
    total = 0.0
    for j in range(2):
        yy = iy + j
        wy = ty if j == 1 else 1.0 - ty
        for i in range(2):
            xx = ix + i
            wx = tx if i == 1 else 1.0 - tx
            if 0 <= xx < n and 0 <= yy < n:
                total += mask[yy, xx] * wx * wy
    return total


@njit(cache=True, nogil=True, fastmath=True)
def _box(row, width, height):  # type: ignore[no-untyped-def]
    """Pixels ``(x0, x1, y0, y1)`` a glyph row can paint (the circle around its rotated box)."""
    reach = max(row[SIZE] * max(width, height) / 2, _MIN_HALF_PX) * 1.4143
    cx, cy = row[X] * width, row[Y] * height
    return (max(0, int(np.floor(cx - reach))), min(width, int(np.ceil(cx + reach)) + 1),
            max(0, int(np.floor(cy - reach))), min(height, int(np.ceil(cy + reach)) + 1))


@njit(cache=True, nogil=True, fastmath=True)
def _draw(genome, canvas, ox, oy, width, height, masks):  # type: ignore[no-untyped-def]
    """Paint the glyphs on ``canvas``, a window of a ``width`` x ``height`` image with its corner at ``(ox, oy)``."""
    side = max(width, height)
    count = masks.shape[0]
    for k in range(genome.shape[0]):
        x0, x1, y0, y1 = _box(genome[k], width, height)
        x0, x1 = max(x0, ox), min(x1, ox + canvas.shape[1])
        y0, y1 = max(y0, oy), min(y1, oy + canvas.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        cx = genome[k, X] * width
        cy = genome[k, Y] * height
        half = max(genome[k, SIZE] * side / 2, _MIN_HALF_PX)
        theta = genome[k, ANGLE] * 2.0 * np.pi
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        mask = masks[min(int(genome[k, GLYPH] * count), count - 1)]
        alpha = genome[k, ALPHA]
        r = genome[k, RED]
        g = genome[k, GREEN]
        b = genome[k, BLUE]
        for py in range(y0, y1):
            dy = py + 0.5 - cy
            for px in range(x0, x1):
                dx = px + 0.5 - cx
                u = dx * cos_t + dy * sin_t  # the pixel in the glyph's own, unrotated frame
                v = -dx * sin_t + dy * cos_t
                if u < -half or u > half or v < -half or v > half:
                    continue
                a = _sample(mask, (u / half + 1.0) * 0.5, (v / half + 1.0) * 0.5) * alpha
                if a > 0.0:
                    keep = 1.0 - a
                    canvas[py - oy, px - ox, 0] = canvas[py - oy, px - ox, 0] * keep + r * a
                    canvas[py - oy, px - ox, 1] = canvas[py - oy, px - ox, 1] * keep + g * a
                    canvas[py - oy, px - ox, 2] = canvas[py - oy, px - ox, 2] * keep + b * a

@njit(cache=True, nogil=True, fastmath=True)
def render_into(genome, background, canvas, masks):  # type: ignore[no-untyped-def]
    fill(canvas, background)
    _draw(genome, canvas, 0, 0, canvas.shape[1], canvas.shape[0], masks)


@njit(cache=True, nogil=True, fastmath=True)
def render_mse(genome, target, background, masks):  # type: ignore[no-untyped-def]
    canvas = np.empty_like(target)
    render_into(genome, background, canvas, masks)
    return canvas_mse(canvas, target)


@njit(cache=True, nogil=True, fastmath=True)
def window_sse(genome, target, background, x0, y0, x1, y1, masks):  # type: ignore[no-untyped-def]
    """Squared error over the window ``x0 <= x < x1``, ``y0 <= y < y1``: only that window is drawn."""
    canvas = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
    fill(canvas, background)
    _draw(genome, canvas, x0, y0, target.shape[1], target.shape[0], masks)
    return window_error(canvas, target, x0, y0)


@njit(cache=True, nogil=True, fastmath=True)
def boxes(rows, width, height):  # type: ignore[no-untyped-def]
    out = np.empty((rows.shape[0], 4), dtype=np.int64)
    for k in range(rows.shape[0]):
        out[k, 0], out[k, 1], out[k, 2], out[k, 3] = _box(rows[k], width, height)
    return out


@njit(cache=True, nogil=True)  # no fastmath: NaN means "draw in full"
def delta_sse(before, after, target, background, max_rows, max_pixels, masks):  # type: ignore[no-untyped-def]
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
        delta += window_sse(after, target, background, x0, y0, x1, y1, masks) - window_sse(before, target, background, x0, y0, x1, y1, masks)
    return delta


def render_image(genome: np.ndarray, width: int, height: int, background: np.ndarray,
                 masks: np.ndarray | None = None) -> np.ndarray:
    canvas = np.empty((height, width, 3), dtype=np.float32)
    render_into(np.ascontiguousarray(genome, dtype=np.float32), background, canvas, atlas() if masks is None else masks)
    return canvas


class GlyphShape(Shape):
    name = "glyph"
    plural = "glyphs"
    cols = COLS
    uses_size = True
    default_elements = 1500
    default_initial = 150

    def render_mse(self, genome: Genome, target: Array, background: Array) -> float:
        return float(render_mse(genome, target, background, atlas()))

    def window_sse(self, genome: Genome, target: Array, background: Array, x0: int, y0: int, x1: int,
                   y1: int) -> float:
        return float(window_sse(genome, target, background, x0, y0, x1, y1, atlas()))

    def boxes(self, rows: Genome, width: int, height: int) -> np.ndarray:
        return boxes(rows, width, height)

    def delta_sse(self, before: Genome, after: Genome, target: Array, background: Array, max_rows: int,
                  max_pixels: int) -> float:
        return float(delta_sse(before, after, target, background, max_rows, max_pixels, atlas()))

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
