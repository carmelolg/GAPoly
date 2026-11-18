# gapoly

`gapoly` = GA + polygons (distribution `carmelolg-gapoly`, import `carmelolg.gapoly`, command `gapoly`).

Genetic algorithm that evolves semi-transparent elements until, stacked, they look like a target photo.
The element is chosen with `--shape`: `polygon` (default), `disc` or `voronoi`.
Built on [GALib](https://github.com/carmelolg/GALib) (`carmelolg.galib`, on PyPI as `carmelolg-galib`).

- **Genome**: float32 array, one row per element, drawn in row order. A polygon has 3-6 vertices and a colour (RGBA), a disc a centre, a radius and a colour, a Voronoi seed a position and a colour.
- **Fitness**: pixel MSE between render and target (lower is better, `maximize = False`).
- **Mutation**: move or reshape an element (a vertex, or the radius of a disc), change colour/alpha, add / remove an element, change overlap order.
- **Crossover**: swap tails of the two element lists.

## Shapes

| `--shape` | Element | Genome row | Default `--elements` |
|---|---|---|---|
| `polygon` | polygon with 3-6 vertices | 17 columns, see below | 400 |
| `disc` | disc, from one pixel up to a group of pixels | 7 columns: `x y radius r g b a` | 2000 |
| `voronoi` | seed of a Voronoi cell: each pixel takes the colour of its nearest seed | 5 columns: `x y r g b` | 1000 |

Each shape is a strategy: `shape.py` defines the abstract `Shape` (draw a genome, error against the target, create and
mutate elements, describe) and `polygons.py`, `discs.py` and `voronoi.py` implement it, each with its genome layout,
Numba rasteriser and mutations. `shapes.py` registers them by name (the values of `--shape`); the problem, the viewer and the
command line only use the interface. `common.py` has `Limits` and the types, `operators.py` the mutation and crossover
shared by every shape, `kernels.py` the Numba helpers they share. A disc is never smaller than the pixel under its
centre, and its radius is a fraction of the longest image side, so discs stay round at any resolution.
Voronoi seeds have no size, alpha or drawing order, so `--start-size` and `--max-size` do nothing for them. The rasteriser finds the nearest seed with a grid, exactly and in time close to the number of pixels: 20000 seeds on a 512x341 image take about 10 ms.

## Polygon genome layout

A genome is a float32 array, one row per polygon, drawn in row order (later rows on top). Values are in [0, 1];
coordinates are fractions of the image side.

```
column:   0 | 1   2 | 3   4 | ...  | 11  12 | 13  14  15  16
meaning:  n | x0  y0  x1  y1  ...    x5  y5 |  r   g   b   a
```

`n` is the number of vertices used (3-6), the rest are ignored. Do not index rows with bare numbers: use the names
in `polygons.py` (`N_VERTICES`, `X(v)`, `Y(v)`, `RED`..`ALPHA`, `vertices`, `color`, `make_row`).
`describe(genome)` prints a genome as text, one line per polygon. The numba kernel (`polygons.py`) reads the
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
