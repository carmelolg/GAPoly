"""Crossover on array genomes, for every shape (mutation is compiled: ``kernels.mutate_rows``). Inputs are never modified."""
from __future__ import annotations

from random import Random

import numpy as np

from .common import Genome, Limits


def crossover(a: Genome, b: Genome, rng: Random, limits: Limits) -> tuple[Genome, Genome]:
    """Swap tails of the two element lists (cut points differ per parent, so lengths change)."""
    cut_a, cut_b = rng.randint(0, len(a)), rng.randint(0, len(b))
    return (_fit(np.concatenate((a[:cut_a], b[cut_b:])), a, limits),
            _fit(np.concatenate((b[:cut_b], a[cut_a:])), b, limits))


def _fit(child: Genome, fallback: Genome, limits: Limits) -> Genome:
    child = child[: limits.max_elements]
    return child if len(child) >= limits.min_elements else fallback.copy()
