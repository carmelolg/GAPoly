"""Evolve semi-transparent polygons, discs or Voronoi points until they look like a target image."""
from .common import Genome, Limits
from .problem import ImageProblem
from .shape import Shape
from .shapes import SHAPES

__all__ = ["SHAPES", "Genome", "ImageProblem", "Limits", "Shape"]
