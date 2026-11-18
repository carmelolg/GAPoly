"""Coarse-to-fine schedule of the element size cap."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from carmelolg.galib import Callback, GenerationStats, GeneticAlgorithm

from ..problem import ImageProblem


class SizeSchedule(Callback):
    """Coarse to fine: the element size cap shrinks (log scale) from its starting value to ``final`` by
    ``until`` (fraction of the run), then stays there. New and mutated elements obey the current cap."""

    def __init__(self, problem: ImageProblem, final: float, total_generations: int, until: float = 0.7) -> None:
        self.problem, self.final = problem, final
        self.start = problem.limits.max_size
        self.span = max(1.0, total_generations * until)

    def on_generation(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> None:
        t = min(1.0, stats.generation / self.span)
        size = self.start * (self.final / self.start) ** t
        self.problem.limits = replace(self.problem.limits, max_size=size)
