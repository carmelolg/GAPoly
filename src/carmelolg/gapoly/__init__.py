"""Evolve semi-transparent polygons until they look like a target image."""
from .problem import PolygonProblem
from .shapes import COLS, Genome, Limits

__all__ = ["COLS", "Genome", "Limits", "PolygonProblem"]
