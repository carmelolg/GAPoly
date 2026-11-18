"""Write the run to disk: PNG snapshots, CSV rows, final image and GIF."""
from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from carmelolg.galib import Callback, GenerationStats, GeneticAlgorithm, Result
from PIL import Image

from ..core.common import Array
from ..problem import ImageProblem


def to_image(array: Array) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")


class Snapshots(Callback):
    """Write the run to disk as it goes, keeping nothing in memory.

    - every ``every`` generations: ``snapshots/gen_NNNNNN.png`` (``output_width`` px);
    - every generation: one row of ``history.csv`` (flushed at each snapshot, so it survives an interruption);
    - at the end: ``final.png``, and ``evolution.gif`` composed from the PNG files on disk, one frame at a time.
    """

    def __init__(self, problem: ImageProblem, out_dir: Path, every: int = 100, output_width: int = 512,
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
        self._csv.writerow(["generation", "best_mse", "mean_mse", "elements"])

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
