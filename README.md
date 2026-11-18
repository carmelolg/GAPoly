# gapoly

`gapoly` = GA + polygons (distribution `carmelolg-gapoly`, import `carmelolg.gapoly`, command `gapoly`).

Genetic algorithm that evolves semi-transparent polygons until, stacked, they look like a target photo.
Built on [GALib](https://github.com/carmelolg/GALib) (`carmelolg.galib`, on PyPI as `carmelolg-galib`).

- **Genome**: float32 array, one row per polygon (3-6 vertices in [0,1] coordinates, RGBA), drawn in row order.
- **Fitness**: pixel MSE between render and target (lower is better, `maximize = False`).
- **Mutation**: move a vertex, shift a polygon, change colour/alpha, add / remove a polygon, change overlap order.
- **Crossover**: swap tails of the two polygon lists.

## Genome layout

A genome is a float32 array, one row per polygon, drawn in row order (later rows on top). Values are in [0, 1];
coordinates are fractions of the image side.

```
column:   0 | 1   2 | 3   4 | ...  | 11  12 | 13  14  15  16
meaning:  n | x0  y0  x1  y1  ...    x5  y5 |  r   g   b   a
```

`n` is the number of vertices used (3-6), the rest are ignored. Do not index rows with bare numbers: use the names
in `shapes.py` (`N_VERTICES`, `X(v)`, `Y(v)`, `RED`..`ALPHA`, `vertices`, `color`, `make_row`).
`describe(genome)` prints a genome as text, one line per polygon. The numba kernel (`fast_render.py`) reads the
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
Polygons shrink coarse-to-fine: from `--start-size` (0.5) to `--max-size` (0.12, fraction of image side) over the first 70% of the run.
The live window lays out its panels by photo orientation: stacked for landscape, side by side for portrait/square.
Rendering and error are one compiled Numba loop (`fast_render.py`, releases the GIL). Fitness runs in parallel with GALib `workers=`: `--workers N` (default all CPUs), `--parallel thread|process` (default thread).
Useful options: `--polygons`, `--population`, `--mutation-rate`, `--size` (working resolution: smaller = faster), `--seed`.

## Tests

```bash
python -m pytest
```

## Ideas not done yet

More shape types, region weights (face counts more), SSIM fitness, greedy hill-climbing baseline.
