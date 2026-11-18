"""GALib Problem: minimise the MSE between the rendered polygons and the target image."""
from __future__ import annotations

from random import Random

import numpy as np
from carmelolg.galib import Problem

from . import operators
from .fast_render import render_mse
from .render import Array, render
from .shapes import COLS, Genome, Limits, random_row


class PolygonProblem(Problem[Genome]):
    maximize = False  # fitness is an error: lower is better

    def __init__(self, target: Array, limits: Limits | None = None, initial_polygons: int = 20,
                 mutations_per_child: float = 3.0) -> None:
        self.target = np.ascontiguousarray(target, dtype=np.float32)  # (height, width, 3), values in [0, 1]
        self.height, self.width = self.target.shape[:2]
        self.limits = limits or Limits()
        self.mutations_per_child = mutations_per_child  # expected polygons edited per child, whatever the genome size
        self.initial_polygons = max(self.limits.min_polygons, min(initial_polygons, self.limits.max_polygons))
        self.background = tuple(float(c) for c in self.target.reshape(-1, 3).mean(axis=0))
        self._background = np.asarray(self.background, dtype=np.float32)
        render_mse(np.zeros((0, COLS), dtype=np.float32), self.target, self._background)  # compile before any thread

    def render(self, genome: Genome, width: int | None = None, height: int | None = None) -> Array:
        return render(genome, width or self.width, height or self.height, self.background)  # type: ignore[arg-type]

    def random_genome(self, rng: Random) -> Genome:
        rows = []
        for _ in range(self.initial_polygons):
            cx, cy = rng.random(), rng.random()
            color = self.target[min(int(cy * self.height), self.height - 1), min(int(cx * self.width), self.width - 1)]
            rows.append(random_row(rng, self.limits, tuple(float(c) for c in color), (cx, cy)))
        return np.stack(rows)

    def fitness(self, genome: Genome) -> float:
        return float(render_mse(genome, self.target, self._background))

    def mutate(self, genome: Genome, rng: Random, rate: float) -> Genome:
        per_polygon = min(1.0, self.mutations_per_child / max(1, len(genome)))
        return operators.mutate(genome, rng, rate, self.limits, per_polygon)

    def crossover(self, a: Genome, b: Genome, rng: Random) -> tuple[Genome, Genome]:
        return operators.crossover(a, b, rng, self.limits)
