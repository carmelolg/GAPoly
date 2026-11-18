"""The Strategy interface: how a genome is read as an image.

A shape knows the layout of one genome row, how to draw a genome (a Numba kernel), how to create and mutate
elements. The problem, the viewer and the command line only talk to this interface, so adding a shape means
writing one module and registering it in ``shapes.py``.
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

    @abstractmethod
    def render_mse(self, genome: Genome, target: Array, background: Array) -> float:
        """Mean squared error between the drawn genome and ``target`` (one compiled loop, no image is built)."""

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
