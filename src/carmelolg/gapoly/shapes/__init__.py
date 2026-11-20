"""The shapes an image can be made of, by name (the values of ``--shape``)."""
from __future__ import annotations

from ..core.shape import Shape
from .discs import DiscShape
from .glyphs import GlyphShape
from .mosaic import MosaicShape
from .polygons import PolygonShape
from .strokes import StrokeShape
from .voronoi import VoronoiShape

POLYGON, DISC, VORONOI = PolygonShape(), DiscShape(), VoronoiShape()
MOSAIC, GLYPH, STROKE = MosaicShape(), GlyphShape(), StrokeShape()
SHAPES: dict[str, Shape] = {shape.name: shape for shape in (POLYGON, DISC, VORONOI, MOSAIC, GLYPH, STROKE)}
