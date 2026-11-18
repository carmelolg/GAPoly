"""The shapes an image can be made of, by name (the values of ``--shape``)."""
from __future__ import annotations

from ..core.shape import Shape
from .discs import DiscShape
from .polygons import PolygonShape
from .voronoi import VoronoiShape

POLYGON, DISC, VORONOI = PolygonShape(), DiscShape(), VoronoiShape()
SHAPES: dict[str, Shape] = {shape.name: shape for shape in (POLYGON, DISC, VORONOI)}
