<table>
  <tr>
    <td valign="middle">
      <img src="https://raw.githubusercontent.com/carmelolg/GAPoly/main/site/assets/logo/logo.svg" alt="GAPoly logo" width="96" height="96">
    </td>
    <td valign="middle">
      <h1>GAPoly</h1>
    </td>
  </tr>
</table>

[![CI](https://github.com/carmelolg/GAPoly/actions/workflows/ci.yml/badge.svg)](https://github.com/carmelolg/GAPoly/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/carmelolg/GAPoly/graph/badge.svg)](https://codecov.io/gh/carmelolg/GAPoly)
[![PyPI](https://img.shields.io/pypi/v/carmelolg-gapoly)](https://pypi.org/project/carmelolg-gapoly/)
[![Python](https://img.shields.io/pypi/pyversions/carmelolg-gapoly)](https://pypi.org/project/carmelolg-gapoly/)
[![Pages](https://github.com/carmelolg/GAPoly/actions/workflows/pages.yml/badge.svg)](https://github.com/carmelolg/GAPoly/actions/workflows/pages.yml)
[![Publish](https://github.com/carmelolg/GAPoly/actions/workflows/publish.yml/badge.svg)](https://github.com/carmelolg/GAPoly/actions/workflows/publish.yml)
[![Gallery](https://img.shields.io/badge/gallery-carmelolg.github.io%2FGAPoly-blue)](https://carmelolg.github.io/GAPoly/)

GAPoly = GA + polygons (distribution `carmelolg-gapoly`, import `carmelolg.gapoly`, command `gapoly`).

Genetic algorithm that evolves elements until, stacked, they look like a target photo.

**Gallery: <https://carmelolg.github.io/GAPoly/>**

The element is chosen with `--shape`: `polygon` (default), `disc`, `voronoi`, `mosaic`, `glyph` or `stroke`.
Built on [GALib](https://carmelolg.github.io/GALib/) (`carmelolg.galib`, on PyPI as `carmelolg-galib`).

- **Genome**: float32 array, one row per element, drawn in row order. A polygon has 3-6 vertices and a colour (RGBA), a disc a centre, a radius and a colour, a Voronoi seed a position and a colour, a mosaic tile a centre, a radius and a colour, a glyph a centre, a size, an angle, a character and a colour, a stroke a Bezier curve (start, control point, end), a brush width and a colour.
- **Fitness**: pixel MSE between render and target (lower is better, `maximize = False`).
- **Mutation**: move or reshape an element (a vertex, or the radius of a disc), change colour/alpha, add / remove an element, change overlap order.
- **Crossover**: swap tails of the two element lists.

## Shapes

| `--shape` | Element | Genome row | Default `--elements` |
|---|---|---|---|
| `polygon` | polygon with 3-6 vertices | 17 columns, see below | 400 |
| `disc` | disc, from one pixel up to a group of pixels | 7 columns: `x y radius r g b a` | 2000 |
| `voronoi` | seed of a Voronoi cell: each pixel takes the colour of its nearest seed | 5 columns: `x y r g b` | 1000 |
| `mosaic` | opaque hexagonal tile snapped to a honeycomb lattice, in several sizes | 6 columns: `x y radius r g b` | 3000 |
| `glyph` | typographic character, rotated and coloured | 9 columns: `x y size angle glyph r g b a` | 1500 |
| `stroke` | brush stroke: quadratic Bezier curve painted with a round brush | 11 columns: `x0 y0 cx cy x1 y1 width r g b a` | 1500 |

Each shape is a strategy: the abstract `Shape` (draw a genome, error against the target, create and mutate elements,
describe) is implemented by `PolygonShape`, `DiscShape`, `VoronoiShape`, `MosaicShape`, `GlyphShape` and `StrokeShape`, and `shapes/__init__.py` registers them by name
(the values of `--shape`). The problem, the callbacks and the command line only use the interface (see "Project layout").
A disc is never smaller than the pixel under its
centre, and its radius is a fraction of the longest image side, so discs stay round at any resolution.
A mosaic tile is an opaque regular hexagon with no alpha. The rasteriser rounds its radius to a size level (steps of the square root of two) and moves its centre to the nearest node of the honeycomb lattice of that level, so tiles of one size fit together with no gaps or overlaps, big tiles cover regions and small ones add detail; a position change smaller than half a cell does not change the image. A glyph is drawn from an atlas of character masks rasterised once with Pillow's font (`GLYPHS` in `shapes/glyphs.py` lists the characters, the exact look depends on the font Pillow finds). A stroke is a curve flattened into short segments; every pixel nearer than half the brush width is covered once, so overlapping parts do not double the opacity. For discs, mosaic tiles, glyphs and strokes `--start-size` / `--max-size` cap the diameter, the glyph box and the box around the three curve points; the brush is at most a third of that.
Voronoi seeds have no size, alpha or drawing order, so `--start-size` and `--max-size` do nothing for them. The rasteriser finds the nearest seed with a grid, exactly and in time close to the number of pixels: 20000 seeds on a 512x341 image take about 10 ms.

## Polygon genome layout

A genome is a float32 array, one row per polygon, drawn in row order (later rows on top). Values are in [0, 1];
coordinates are fractions of the image side.

```
column:   0 | 1   2 | 3   4 | ...  | 11  12 | 13  14  15  16
meaning:  n | x0  y0  x1  y1  ...    x5  y5 |  r   g   b   a
```

`n` is the number of vertices used (3-6), the rest are ignored. Do not index rows with bare numbers: use the names
in `shapes/polygons.py` (`N_VERTICES`, `X(v)`, `Y(v)`, `RED`..`ALPHA`, `vertices`, `color`, `make_row`).
`describe(genome)` prints a genome as text, one line per polygon. The numba kernel (in the same module) reads the
same layout; `test_row_layout_is_the_documented_one` fails if the columns change.

## Setup

```bash
pip install carmelolg-gapoly           # from PyPI
pip install "carmelolg-gapoly[live]"   # plus the live matplotlib window
```

From a clone (development):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[live,dev]"
```

## Project layout

```
src/carmelolg/gapoly/
  problem.py        ImageProblem: the GALib Problem (fitness = MSE against the target), works with any Shape
  core/             what every shape shares
    shape.py          Shape, the Strategy interface
    common.py         Limits, clamp, Genome / Row / Array types
    operators.py      mutation skeleton and crossover used by every shape
    kernels.py        Numba helpers (fill, MSE)
  shapes/           one module per way of reading the genome: layout, Numba rasteriser, mutations, Shape class
    __init__.py       SHAPES, the registry (values of --shape)
    polygons.py  discs.py  voronoi.py  mosaic.py  glyphs.py  strokes.py
  callbacks/        GALib callbacks: snapshots.py (PNG, CSV, GIF), live.py (matplotlib), schedule.py (size cap)
  cli/              app.py (arguments, help text, main), help_format.py (colours)
tests/test_gapoly.py
```

To add a shape: write a module in `shapes/` with the layout, kernel, operators and a `Shape` subclass, and add it to
`shapes/__init__.py`.

## Run

```bash
python -m carmelolg.gapoly photo.jpg --generations 5000 --live          # live window
python -m carmelolg.gapoly photo.jpg --snapshot-every 1 --generations 300  # one PNG per generation
```

Output in `out/<name>/`: `snapshots/gen_*.png`, `final.png`, `evolution.gif`, `history.csv`.
Elements shrink coarse-to-fine: from `--start-size` (0.5) to `--max-size` (0.12, fraction of image side) over the first 70% of the run.
The live window lays out its panels by photo orientation: stacked for landscape, side by side for portrait/square.
Rendering and error are one compiled Numba loop that releases the GIL, so fitness runs in parallel with GALib `workers=` in a thread pool: `--workers N` (default all CPUs).
Useful options: `--shape`, `--elements`, `--population`, `--mutation-rate`, `--size` (working resolution: smaller = faster), `--seed`.

## Tests

```bash
python -m pytest
```

## Ideas not done yet

Region weights (face counts more), SSIM fitness, greedy hill-climbing baseline.
