"""GALib Problem: minimise the MSE between the drawn elements and the target image."""
from __future__ import annotations

from random import Random

import numpy as np
from carmelolg.galib import Problem

from .core import operators
from .core.common import Array, Genome, Limits
from .core.shape import Shape
from .shapes import POLYGON


class ImageProblem(Problem[Genome]):
    """Approximate ``target`` with elements drawn by a ``shape`` strategy (polygons by default)."""

    maximize = False  # fitness is an error: lower is better

    def __init__(self, target: Array, limits: Limits | None = None, initial_elements: int = 20,
                 mutations_per_child: float = 3.0, shape: Shape = POLYGON) -> None:
        self.shape = shape
        self.target = np.ascontiguousarray(target, dtype=np.float32)  # (height, width, 3), values in [0, 1]
        self.height, self.width = self.target.shape[:2]
        self.limits = limits or Limits()
        self.mutations_per_child = mutations_per_child  # expected elements edited per child, whatever the genome size
        self.initial_elements = max(self.limits.min_elements, min(initial_elements, self.limits.max_elements))
        self.background = tuple(float(c) for c in self.target.reshape(-1, 3).mean(axis=0))
        self._background = np.asarray(self.background, dtype=np.float32)
        shape.render_mse(shape.empty_genome(), self.target, self._background)  # compile before any thread

    def render(self, genome: Genome, width: int | None = None, height: int | None = None) -> Array:
        return self.shape.render(genome, width or self.width, height or self.height, self.background)  # type: ignore[arg-type]

    def random_genome(self, rng: Random) -> Genome:
        rows = []
        for _ in range(self.initial_elements):
            cx, cy = rng.random(), rng.random()
            color = self.target[min(int(cy * self.height), self.height - 1), min(int(cx * self.width), self.width - 1)]
            rows.append(self.shape.random_row(rng, self.limits, tuple(float(c) for c in color), (cx, cy)))
        return np.stack(rows)

    def fitness(self, genome: Genome) -> float:
        return self.shape.render_mse(genome, self.target, self._background)

    def mutate(self, genome: Genome, rng: Random, rate: float) -> Genome:
        per_element = min(1.0, self.mutations_per_child / max(1, len(genome)))
        return self.shape.mutate(genome, rng, rate, self.limits, per_element)

    def crossover(self, a: Genome, b: Genome, rng: Random) -> tuple[Genome, Genome]:
        return operators.crossover(a, b, rng, self.limits)
