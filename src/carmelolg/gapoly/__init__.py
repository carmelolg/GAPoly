"""Evolve polygons, discs, Voronoi points, mosaic tiles, glyphs or brush strokes until they look like a target image."""
from .core.common import Genome, Limits
from .core.shape import Shape
from .problem import ImageProblem
from .shapes import SHAPES

__all__ = ["SHAPES", "Genome", "ImageProblem", "Limits", "Shape"]
