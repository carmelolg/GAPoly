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

**GA + polygons.** A genetic algorithm, built on [GALib](https://carmelolg.github.io/GALib/), that evolves
thousands of simple elements (polygons, discs, Voronoi cells, tiles, glyphs or brush strokes) until, stacked,
they look like a target photo.

**See what it makes in the [gallery](https://carmelolg.github.io/GAPoly/).**

| | |
|---|---|
| Distribution | `carmelolg-gapoly` |
| Import | `carmelolg.gapoly` |
| Command | `gapoly` |

## Quick start

```bash
pip install carmelolg-gapoly           # from PyPI
pip install "carmelolg-gapoly[live]"   # plus the live matplotlib window
```

```bash
gapoly photo.jpg --generations 5000 --live             # live window
gapoly photo.jpg --shape voronoi --elements 1000       # pick the kind of element
gapoly photo.jpg --snapshot-every 1 --generations 300  # one PNG per generation
```

`python -m carmelolg.gapoly` works as well as `gapoly`. Everything is written to `out/<name>/`:

| File | What it is |
|---|---|
| `final.png` | the result |
| `evolution.gif` | how the image evolved |
| `history.csv` | the error per generation |
| `snapshots/gen_*.png` | one image per snapshot (with `--snapshot-every`) |

### Useful options

| Option | Effect |
|---|---|
| `--shape` | kind of element, see [Shapes](#shapes) |
| `--elements` | number of elements in the image |
| `--population`, `--mutation-rate` | genetic algorithm parameters |
| `--size` | working resolution: smaller is faster |
| `--workers N` | threads for the fitness (default: all CPUs) |
| `--seed` | reproducible run |
| `--start-size`, `--max-size` | size of the elements, see below |

Elements shrink coarse to fine: from `--start-size` (0.5) to `--max-size` (0.12), as fractions of the image side,
over the first 70% of the run. Rendering and error are one compiled Numba loop that releases the GIL, so the
fitness runs in parallel with GALib's `workers=`. The live window stacks its panels for landscape photos and
puts them side by side for portrait or square ones.

## How it works

| Part | What happens |
|---|---|
| **Genome** | a float32 array, one row per element, drawn in row order (later rows on top) |
| **Fitness** | pixel MSE between the render and the target (lower is better, `maximize = False`) |
| **Mutation** | move or reshape an element, change colour or alpha, add or remove an element, change the overlap order |
| **Crossover** | swap the tails of the two element lists |

## Shapes

The element is chosen with `--shape`.

| `--shape` | Element | Genome row | Default `--elements` |
|---|---|---|---|
| `polygon` (default) | polygon with 3-6 vertices | 17 columns, see [below](#polygon-genome-layout) | 400 |
| `disc` | disc, from one pixel up to a group of pixels | 7 columns: `x y radius r g b a` | 2000 |
| `voronoi` | seed of a Voronoi cell: each pixel takes the colour of its nearest seed | 5 columns: `x y r g b` | 1000 |
| `mosaic` | opaque hexagonal tile snapped to a honeycomb lattice, in several sizes | 6 columns: `x y radius r g b` | 3000 |
| `glyph` | typographic character, rotated and coloured | 9 columns: `x y size angle glyph r g b a` | 1500 |
| `stroke` | brush stroke: quadratic Bezier curve painted with a round brush | 11 columns: `x0 y0 cx cy x1 y1 width r g b a` | 1500 |

Details of each shape:

- **Disc**: never smaller than the pixel under its centre. Its radius is a fraction of the longest image side, so
  discs stay round at any resolution.
- **Voronoi**: seeds have no size, alpha or drawing order, so `--start-size` and `--max-size` do nothing for them.
  The rasteriser finds the nearest seed with a grid, exactly and in time close to the number of pixels: 20000 seeds
  on a 512x341 image take about 10 ms.
- **Mosaic**: an opaque regular hexagon with no alpha. The radius is rounded to a size level (steps of the square
  root of two) and the centre moves to the nearest node of that level's honeycomb lattice. Tiles of one size fit
  together with no gaps or overlaps: big tiles cover regions, small ones add detail. A position change smaller
  than half a cell does not change the image.
- **Glyph**: drawn from an atlas of character masks, rasterised once with Pillow's font. `GLYPHS` in
  `shapes/glyphs.py` lists the characters; the exact look depends on the font Pillow finds.
- **Stroke**: a curve flattened into short segments. Every pixel nearer than half the brush width is covered
  once, so overlapping parts do not double the opacity.

For discs, mosaic tiles, glyphs and strokes, `--start-size` and `--max-size` cap the diameter, the glyph box and
the box around the three curve points. The brush is at most a third of that.

### Polygon genome layout

Values are in [0, 1]; coordinates are fractions of the image side.

```
column:   0 | 1   2 | 3   4 | ...  | 11  12 | 13  14  15  16
meaning:  n | x0  y0  x1  y1  ...    x5  y5 |  r   g   b   a
```

`n` is the number of vertices used (3-6); the rest are ignored. Do not index rows with bare numbers: use the names
in `shapes/polygons.py` (`N_VERTICES`, `X(v)`, `Y(v)`, `RED`..`ALPHA`, `vertices`, `color`, `make_row`).
`describe(genome)` prints a genome as text, one line per polygon. The Numba kernel in the same module reads the
same layout, and `test_row_layout_is_the_documented_one` fails if the columns change.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[live,dev]"
python -m pytest
```

### Project layout

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
tests/
site/               the gallery (published on GitHub Pages)
tools/              build_site.py, optimize_works.py
```

Each shape is a strategy: the abstract `Shape` (draw a genome, error against the target, create and mutate
elements, describe) is implemented by `PolygonShape`, `DiscShape`, `VoronoiShape`, `MosaicShape`, `GlyphShape` and
`StrokeShape`. The problem, the callbacks and the command line only use that interface.

**To add a shape**: write a module in `shapes/` with the layout, kernel, operators and a `Shape` subclass, then
register it in `shapes/__init__.py`.

## Ideas not done yet

Region weights (a face counts more), SSIM fitness, a greedy hill-climbing baseline.
