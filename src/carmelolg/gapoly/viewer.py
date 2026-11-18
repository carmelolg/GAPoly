"""Callbacks that show or save the best image while the algorithm runs."""
from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from carmelolg.galib import Callback, GenerationStats, GeneticAlgorithm, Result
from PIL import Image

from .problem import PolygonProblem
from .render import to_image
from .shapes import empty_genome


class SizeSchedule(Callback):
    """Coarse to fine: the polygon size cap shrinks (log scale) from its starting value to ``final`` by
    ``until`` (fraction of the run), then stays there. New and mutated polygons obey the current cap."""

    def __init__(self, problem: PolygonProblem, final: float, total_generations: int, until: float = 0.7) -> None:
        self.problem, self.final = problem, final
        self.start = problem.limits.max_size
        self.span = max(1.0, total_generations * until)

    def on_generation(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> None:
        t = min(1.0, stats.generation / self.span)
        size = self.start * (self.final / self.start) ** t
        self.problem.limits = replace(self.problem.limits, max_size=size)


class Snapshots(Callback):
    """Write the run to disk as it goes, keeping nothing in memory.

    - every ``every`` generations: ``snapshots/gen_NNNNNN.png`` (``output_width`` px);
    - every generation: one row of ``history.csv`` (flushed at each snapshot, so it survives an interruption);
    - at the end: ``final.png``, and ``evolution.gif`` composed from the PNG files on disk, one frame at a time.
    """

    def __init__(self, problem: PolygonProblem, out_dir: Path, every: int = 100, output_width: int = 512,
                 frame_width: int = 256) -> None:
        self.problem, self.out_dir, self.every = problem, out_dir, max(1, every)
        self.output_width, self.frame_width = output_width, frame_width
        self.snapshots = out_dir / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self._csv_file: TextIO | None = None
        self._csv: Any = None

    def _image(self, genome: Any, width: int) -> Image.Image:
        height = round(width * self.problem.height / self.problem.width)
        return to_image(self.problem.render(genome, width, height))

    def on_start(self, engine: GeneticAlgorithm[Any]) -> None:
        self._csv_file = open(self.out_dir / "history.csv", "w", newline="")  # noqa: SIM115 (closed in on_end)
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["generation", "best_mse", "mean_mse", "polygons"])

    def on_generation(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> None:
        self._csv.writerow([stats.generation, stats.best_fitness, stats.mean_fitness, len(stats.best_genome)])
        if stats.generation % self.every:
            return
        self._image(stats.best_genome, self.output_width).save(self.snapshots / f"gen_{stats.generation:06d}.png")
        if self._csv_file is not None:
            self._csv_file.flush()

    def on_end(self, result: Result[Any], engine: GeneticAlgorithm[Any]) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        self._image(result.best.genome, self.output_width).save(self.out_dir / "final.png")
        self._write_gif()

    def _frames(self) -> Iterator[Image.Image]:
        height = round(self.frame_width * self.problem.height / self.problem.width)
        for path in [*sorted(self.snapshots.glob("gen_*.png")), self.out_dir / "final.png"]:
            with Image.open(path) as image:
                yield image.convert("RGB").resize((self.frame_width, height), Image.LANCZOS)

    def _write_gif(self) -> None:
        frames = self._frames()
        first = next(frames)
        first.save(self.out_dir / "evolution.gif", save_all=True, append_images=frames, duration=120, loop=0)


class LiveView(Callback):
    """matplotlib window: target | best render | error curve. Needs ``pip install matplotlib``."""

    def __init__(self, problem: PolygonProblem, every: int = 10) -> None:
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
        self.image = self.a_best.imshow(problem.render(empty_genome()))
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
        self.a_best.set_title(f"generation {stats.generation}  mse {stats.best_fitness:.5f}  polygons {len(stats.best_genome)}")
        self.line.set_data(self.x, self.y)
        self.a_curve.relim()
        self.a_curve.autoscale_view()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def on_end(self, result: Result[Any], engine: GeneticAlgorithm[Any]) -> None:
        self.plt.ioff()
        self.plt.show()
