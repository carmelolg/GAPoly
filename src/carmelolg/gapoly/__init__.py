"""Evolve semi-transparent polygons, discs or Voronoi points until they look like a target image."""
from .kinds import KINDS, Kind
from .limits import Limits
from .problem import ImageProblem, PolygonProblem
from .shapes import COLS, Genome

__all__ = ["COLS", "KINDS", "Genome", "ImageProblem", "Kind", "Limits", "PolygonProblem"]
