"""Mutation and crossover on array genomes. Inputs are never modified."""
from __future__ import annotations

import math
from random import Random

import numpy as np

from .shapes import ALPHA, GREEN, N_VERTICES, RED, Genome, Limits, X, Y, clamp, extent, random_row


def _selected(rng: Random, count: int, probability: float) -> list[int]:
    """Indices in ``range(count)`` picked independently with ``probability`` (geometric jumps: one draw per pick)."""
    if probability >= 1.0:
        return list(range(count))
    if probability <= 0.0:
        return []
    log_miss = math.log1p(-probability)
    picked, i = [], -1
    while True:
        i += 1 + int(math.log(1.0 - rng.random()) / log_miss)
        if i >= count:
            return picked
        picked.append(i)


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


def mutate(genome: Genome, rng: Random, rate: float, limits: Limits, per_polygon: float | None = None) -> Genome:
    """``per_polygon``: probability that each polygon changes shape / colour (default ``rate``).
    ``rate``: probability, per genome, of each add / remove / reorder operation."""
    g = genome.copy()
    p = rate if per_polygon is None else per_polygon
    for k in _selected(rng, len(g), p):
        if rng.random() < 0.7:
            move_vertex(g[k], rng, limits)
        else:
            shift_polygon(g[k], rng)
    for k in _selected(rng, len(g), p):
        change_color(g[k], rng, limits)
    if rng.random() < rate and len(g) < limits.max_polygons:
        g = np.insert(g, rng.randrange(len(g) + 1), random_row(rng, limits), axis=0)
    if rng.random() < rate and len(g) > limits.min_polygons:
        g = np.delete(g, rng.randrange(len(g)), axis=0)
    if rng.random() < rate and len(g) > 1:  # change overlap order
        source = rng.randrange(len(g))
        row = g[source].copy()
        g = np.insert(np.delete(g, source, axis=0), rng.randrange(len(g)), row, axis=0)
    return g


def crossover(a: Genome, b: Genome, rng: Random, limits: Limits) -> tuple[Genome, Genome]:
    """Swap tails of the two polygon lists (cut points differ per parent, so lengths change)."""
    cut_a, cut_b = rng.randint(0, len(a)), rng.randint(0, len(b))
    return (_fit(np.concatenate((a[:cut_a], b[cut_b:])), a, limits),
            _fit(np.concatenate((b[:cut_b], a[cut_a:])), b, limits))


def _fit(child: Genome, fallback: Genome, limits: Limits) -> Genome:
    child = child[: limits.max_polygons]
    return child if len(child) >= limits.min_polygons else fallback.copy()
