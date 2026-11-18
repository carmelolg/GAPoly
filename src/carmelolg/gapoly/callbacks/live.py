"""matplotlib window shown while the algorithm runs."""
from __future__ import annotations

from typing import Any

from carmelolg.galib import Callback, GenerationStats, GeneticAlgorithm, Result

from ..problem import ImageProblem


class LiveView(Callback):
    """matplotlib window: target | best render | error curve. Needs ``pip install matplotlib``."""

    def __init__(self, problem: ImageProblem, every: int = 10) -> None:
        import matplotlib.pyplot as plt

        self.plt, self.problem, self.every = plt, problem, max(1, every)
        plt.ion()
        # Landscape photo: panels stacked in a column (wide images fit). Portrait or square: side by side.
        landscape = problem.width > problem.height
        aspect = problem.height / problem.width
        if landscape:
            self.fig, (a_target, self.a_best, self.a_curve) = plt.subplots(3, 1, figsize=(6, 2 * 6 * aspect + 3))
        else:
            self.fig, (a_target, self.a_best, self.a_curve) = plt.subplots(1, 3, figsize=(3 * 4.2, 4.2 * aspect + 1))
        a_target.imshow(problem.target)
        a_target.set_title("target")
        self.a_best.axis("off")
        a_target.axis("off")
        self.image = self.a_best.imshow(problem.render(problem.shape.empty_genome()))
        self.a_curve.set_xlabel("generation")
        self.a_curve.set_ylabel("MSE")
        (self.line,) = self.a_curve.plot([], [])
        self.x: list[int] = []
        self.y: list[float] = []

    def on_generation(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> None:
        self.x.append(stats.generation)
        self.y.append(stats.best_fitness)
        if stats.generation % self.every:
            return
        self.image.set_data(self.problem.render(stats.best_genome))
        plural = self.problem.shape.plural
        self.a_best.set_title(f"generation {stats.generation}  mse {stats.best_fitness:.5f}  {plural} {len(stats.best_genome)}")
        self.line.set_data(self.x, self.y)
        self.a_curve.relim()
        self.a_curve.autoscale_view()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def on_end(self, result: Result[Any], engine: GeneticAlgorithm[Any]) -> None:
        self.plt.ioff()
        self.plt.show()
