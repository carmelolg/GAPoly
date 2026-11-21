"""The Strategy interface: how a genome is read as an image.

A shape knows the layout of one genome row, how to draw a genome (a Numba kernel), how to create and mutate
elements. The problem, the callbacks and the command line only talk to this interface, so adding a shape means
writing one module and registering it in ``shapes/__init__.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random
from typing import ClassVar

import numpy as np

from .common import Array, Genome, Limits, Row


class Shape(ABC):
    name: ClassVar[str]  # value of ``--shape``
    plural: ClassVar[str]  # for messages: "polygons"
    cols: ClassVar[int]  # columns of one genome row
    uses_size: ClassVar[bool]  # whether ``Limits.max_size`` (and the size schedule) mean something
    default_elements: ClassVar[int]  # default for ``--elements``
    default_initial: ClassVar[int]  # elements each individual starts with
    incremental: ClassVar[bool] = True  # a changed element only changes the pixels in its box (False: Voronoi)

    @abstractmethod
    def render_mse(self, genome: Genome, target: Array, background: Array) -> float:
        """Mean squared error between the drawn genome and ``target`` (one compiled loop, no image is built)."""

    def window_sse(self, genome: Genome, target: Array, background: Array, x0: int, y0: int, x1: int, y1: int) -> float:
        """Sum of squared errors of the drawn genome over the pixels ``x0 <= x < x1``, ``y0 <= y < y1``, drawing only
        that window. A shape with ``incremental = True`` implements it, with ``boxes``."""
        raise NotImplementedError(f"{type(self).__name__} cannot be evaluated by windows")

    def delta_sse(self, before: Genome, after: Genome, target: Array, background: Array, max_rows: int,
                  max_pixels: int) -> float:
        """Squared error of ``after`` minus that of ``before``, two genomes that differ in a few rows, computed in one
        compiled call that redraws only the windows around the differing rows. NaN when more than ``max_rows`` rows
        differ or the windows cover more than ``max_pixels``: the caller then draws ``after`` in full."""
        raise NotImplementedError(f"{type(self).__name__} cannot be evaluated by windows")

    def boxes(self, rows: Genome, width: int, height: int) -> np.ndarray:
        """The pixels each row paints, as an int64 array of ``(x0, x1, y0, y1)``, one per row, clipped to the image.
        Empty when ``x1 <= x0`` or ``y1 <= y0``."""
        raise NotImplementedError(f"{type(self).__name__} cannot be evaluated by windows")

    @abstractmethod
    def render_image(self, genome: Genome, width: int, height: int, background: Array) -> Array:
        """The drawn genome as a float32 image of shape (height, width, 3)."""

    @abstractmethod
    def random_row(self, rng: Random, limits: Limits, rgb: tuple[float, float, float] | None = None,
                   center: tuple[float, float] | None = None) -> Row:
        """A new element at ``center`` (random if omitted) with colour ``rgb`` (random if omitted)."""

    @abstractmethod
    def mutate(self, genome: Genome, rng: Random, rate: float, limits: Limits,
               per_element: float | None = None) -> Genome:
        """A mutated copy: ``per_element`` is the probability that each element changes, ``rate`` the probability of
        each add / remove / reorder."""

    @abstractmethod
    def describe(self, genome: Genome) -> str:
        """One line per element, in drawing order: handy when debugging."""

    def empty_genome(self) -> Genome:
        return np.zeros((0, self.cols), dtype=np.float32)

    def render(self, genome: Genome, width: int, height: int, background: tuple[float, float, float]) -> Array:
        return self.render_image(genome, width, height, np.asarray(background, dtype=np.float32))
