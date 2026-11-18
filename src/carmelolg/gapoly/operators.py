"""Mutation and crossover on array genomes, for every shape. Inputs are never modified."""
from __future__ import annotations

import math
from collections.abc import Callable
from random import Random

import numpy as np

from .common import Genome, Limits


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


def mutate_rows(genome: Genome, rng: Random, rate: float, limits: Limits, per_row: float | None,
                edit_shape: Callable[[np.ndarray, Random, Limits], None],
                edit_color: Callable[[np.ndarray, Random, Limits], None],
                new_row: Callable[[Random, Limits], np.ndarray], reorder: bool = True) -> Genome:
    """The mutation shared by every shape.

    ``per_row``: probability that each element changes shape / colour (default ``rate``).
    ``rate``: probability, per genome, of each add / remove / reorder operation. ``edit_shape``, ``edit_color``
    change one row in place (a private copy); ``new_row`` makes a fresh element. ``reorder=False`` for shapes where
    the drawing order does not matter (Voronoi points).
    """
    g = genome.copy()
    p = rate if per_row is None else per_row
    for k in _selected(rng, len(g), p):
        edit_shape(g[k], rng, limits)
    for k in _selected(rng, len(g), p):
        edit_color(g[k], rng, limits)
    if rng.random() < rate and len(g) < limits.max_elements:
        g = np.insert(g, rng.randrange(len(g) + 1), new_row(rng, limits), axis=0)
    if rng.random() < rate and len(g) > limits.min_elements:
        g = np.delete(g, rng.randrange(len(g)), axis=0)
    if reorder and rng.random() < rate and len(g) > 1:  # change overlap order
        source = rng.randrange(len(g))
        row = g[source].copy()
        g = np.insert(np.delete(g, source, axis=0), rng.randrange(len(g)), row, axis=0)
    return g


def crossover(a: Genome, b: Genome, rng: Random, limits: Limits) -> tuple[Genome, Genome]:
    """Swap tails of the two element lists (cut points differ per parent, so lengths change)."""
    cut_a, cut_b = rng.randint(0, len(a)), rng.randint(0, len(b))
    return (_fit(np.concatenate((a[:cut_a], b[cut_b:])), a, limits),
            _fit(np.concatenate((b[:cut_b], a[cut_a:])), b, limits))


def _fit(child: Genome, fallback: Genome, limits: Limits) -> Genome:
    child = child[: limits.max_elements]
    return child if len(child) >= limits.min_elements else fallback.copy()
