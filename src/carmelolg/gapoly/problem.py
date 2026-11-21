"""GALib Problem: minimise the MSE between the drawn elements and the target image."""
from __future__ import annotations

from random import Random

import numpy as np
from carmelolg.galib import Individual, Problem

from .core import operators
from .core.common import Array, Genome, Limits
from .core.kernels import merge_boxes, unmatched_rows
from .core.shape import Shape
from .shapes import POLYGON

MAX_CHANGED_ROWS = 64  # a child that differs from its parent in more rows than this is drawn in full
MAX_REDRAWN = 0.5  # ... and so is one whose changed windows cover more than this fraction of the image


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

    def fitness_from_parent(self, genome: Genome, parent: Individual[Genome]) -> float:
        """The error of a mutated child from its parent's, drawing only the pixels the changes can touch.

        Elements the child shares with the parent, in the same order, paint the same pixels. So the error differs
        only inside the boxes of the elements that were added, removed or edited (before and after): the two
        genomes are drawn in those windows alone and the difference of the squared errors is added to the parent's.
        """
        shape = self.shape
        if not shape.incremental:
            return self.fitness(genome)
        before, after = parent.genome, genome
        old_rows, new_rows, n_old, n_new = unmatched_rows(before, after, MAX_CHANGED_ROWS)
        if n_old < 0:
            return self.fitness(genome)
        if n_old == 0 and n_new == 0:
            return parent.fitness
        areas = np.concatenate((shape.boxes(before[old_rows[:n_old]], self.width, self.height),
                                shape.boxes(after[new_rows[:n_new]], self.width, self.height)))
        windows = merge_boxes(areas)
        pixels = int(((windows[:, 1] - windows[:, 0]) * (windows[:, 3] - windows[:, 2])).sum())
        if pixels > MAX_REDRAWN * self.width * self.height:
            return self.fitness(genome)
        delta = 0.0
        for x0, x1, y0, y1 in windows:
            delta += (shape.window_sse(after, self.target, self._background, x0, y0, x1, y1)
                      - shape.window_sse(before, self.target, self._background, x0, y0, x1, y1))
        return parent.fitness + delta / self.target.size

    def mutate(self, genome: Genome, rng: Random, rate: float) -> Genome:
        per_element = min(1.0, self.mutations_per_child / max(1, len(genome)))
        return self.shape.mutate(genome, rng, rate, self.limits, per_element)

    def crossover(self, a: Genome, b: Genome, rng: Random) -> tuple[Genome, Genome]:
        return operators.crossover(a, b, rng, self.limits)
