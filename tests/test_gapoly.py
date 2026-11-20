import itertools
import re
from random import Random

import numpy as np
import pytest
from carmelolg.galib import GeneticAlgorithm
from PIL import Image, ImageDraw

from carmelolg.gapoly import SHAPES, ImageProblem, Limits, Shape
from carmelolg.gapoly.core import operators
from carmelolg.gapoly.shapes import (
    DISC,
    GLYPH,
    MOSAIC,
    POLYGON,
    STROKE,
    VORONOI,
    discs,
    glyphs,
    mosaic,
    polygons,
    strokes,
    voronoi,
)
from carmelolg.gapoly.shapes.polygons import (
    ALPHA,
    COLS,
    N_VERTICES,
    RED,
    X,
    Y,
    color,
    describe,
    extent,
    make_row,
    random_row,
    vertices,
)

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def render(genome, width, height, background, shape=POLYGON):
    return shape.render(genome, width, height, background)


def gradient_target(size: int = 24) -> np.ndarray:
    x = np.linspace(0, 1, size, dtype=np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = x[None, :]
    img[..., 2] = x[:, None]
    return img


def test_render_opaque_and_blended():
    red = np.stack([make_row(SQUARE, (1.0, 0.0, 0.0, 1.0))])
    assert render(red, 8, 8, (0.0, 0.0, 0.0))[4, 4].tolist() == [1.0, 0.0, 0.0]
    half = np.stack([make_row(SQUARE, (1.0, 0.0, 0.0, 0.5))])
    assert render(half, 8, 8, (0.0, 0.0, 0.0))[4, 4, 0] == pytest.approx(0.5)
    assert render(POLYGON.empty_genome(), 4, 4, (0.2, 0.4, 0.6))[0, 0].tolist() == pytest.approx([0.2, 0.4, 0.6])


def pillow_reference(genome, size, background):
    canvas = np.empty((size, size, 3), dtype=np.float32)
    canvas[:] = background
    for r in genome:
        xs, ys = vertices(r)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).polygon([(x * size, y * size) for x, y in zip(xs, ys, strict=True)], fill=255)
        alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0 * color(r)[3]
        canvas = canvas * (1 - alpha) + color(r)[:3] * alpha
    return canvas


def test_numba_rasteriser_matches_pillow():
    limits = Limits(max_size=0.4)
    rng = Random(5)
    genome = np.stack([random_row(rng, limits) for _ in range(30)])
    ours = render(genome, 64, 64, (0.5, 0.5, 0.5))
    reference = pillow_reference(genome, 64, (0.5, 0.5, 0.5))
    assert np.abs(ours - reference).mean() < 0.01  # same shapes, only edge pixels differ


def test_fitness_zero_for_perfect_and_better_when_closer():
    problem = ImageProblem(np.zeros((8, 8, 3), dtype=np.float32))
    assert problem.fitness(POLYGON.empty_genome()) == 0.0
    far = np.stack([make_row(SQUARE, (1.0, 1.0, 1.0, 0.9))])
    near = np.stack([make_row(SQUARE, (1.0, 1.0, 1.0, 0.1))])
    assert problem.fitness(near) < problem.fitness(far)


def test_fitness_equals_numpy_mse_of_the_render():
    problem = ImageProblem(gradient_target(), Limits(max_size=0.5), initial_elements=10)
    genome = problem.random_genome(Random(3))
    expected = float(np.mean((problem.render(genome) - problem.target) ** 2))
    assert problem.fitness(genome) == pytest.approx(expected, rel=1e-4)


def test_operators_keep_genome_valid_and_inputs_untouched():
    limits = Limits(max_elements=12)
    problem = ImageProblem(gradient_target(), limits, initial_elements=6)
    rng = Random(1)
    a, b = problem.random_genome(rng), problem.random_genome(rng)
    a0, b0 = a.copy(), b.copy()
    polygons.mutate(a, rng, 1.0, limits)
    operators.crossover(a, b, rng, limits)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)
    for _ in range(300):
        a = polygons.mutate(a, rng, 0.3, limits)
        a, b = operators.crossover(a, b, rng, limits)
        for genome in (a, b):
            assert genome.shape[1] == COLS and genome.dtype == np.float32
            assert limits.min_elements <= len(genome) <= limits.max_elements
            for r in genome:
                xs, ys = vertices(r)
                assert limits.min_vertices <= len(xs) <= limits.max_vertices
                assert np.all((xs >= 0) & (xs <= 1) & (ys >= 0) & (ys <= 1))
                assert np.all((color(r) >= 0) & (color(r) <= 1))
                assert limits.min_alpha <= r[ALPHA] <= limits.max_alpha
                assert extent(r) <= limits.max_size + 1e-6


def run(seed: int, **kwargs):
    problem = ImageProblem(gradient_target(), Limits(max_elements=20), initial_elements=5)
    return GeneticAlgorithm(problem, population_size=16, max_generations=40, mutation_rate=0.1, seed=seed,
                            diversity=None, **kwargs).run()


def test_evolution_improves_and_is_reproducible():
    first = run(7)
    assert first.history[-1].best_fitness < first.history[0].best_fitness
    assert run(7).best.fitness == first.best.fitness


def test_threads_give_the_same_result_as_serial():
    assert run(7, workers=3, parallel="thread").best.fitness == run(7).best.fitness


def test_coloured_help_is_the_plain_help_plus_escape_codes(capsys, monkeypatch):
    from carmelolg.gapoly.cli import main

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    with pytest.raises(SystemExit):
        main(["-h"])
    plain = capsys.readouterr().out
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    with pytest.raises(SystemExit):
        main(["-h"])
    coloured = capsys.readouterr().out
    assert "\033[" in coloured and "\033[" not in plain
    assert re.sub(r"\033\[[0-9;]*m", "", coloured) == plain


def test_row_layout_is_the_documented_one():
    """If this fails, someone changed the columns: update the diagram in polygons.py and the numba kernel too."""
    r = make_row([(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)], (0.7, 0.8, 0.9, 0.5))
    assert COLS == 17 and len(r) == COLS
    assert r[N_VERTICES] == 3 and (N_VERTICES, X(0), Y(0), X(1), Y(1), X(5), Y(5)) == (0, 1, 2, 3, 4, 11, 12)
    assert (RED, ALPHA) == (13, 16)
    assert r[[X(0), Y(0), X(2), Y(2)]].tolist() == pytest.approx([0.1, 0.2, 0.5, 0.6])
    assert color(r).tolist() == pytest.approx([0.7, 0.8, 0.9, 0.5])
    assert not r[X(3) : RED].any()  # unused vertices stay zero


def test_describe_prints_one_line_per_polygon():
    text = describe(np.stack([make_row([(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)], (1.0, 0.0, 0.0, 0.5))]))
    assert text == "#0 3 vertices (0.10,0.20) (0.30,0.40) (0.50,0.60) rgba(1.00,0.00,0.00,0.50)"


def test_history_genome_every_keeps_stats_but_drops_most_genomes():
    problem = ImageProblem(gradient_target(), Limits(max_elements=20), initial_elements=5)
    result = GeneticAlgorithm(problem, population_size=10, max_generations=25, seed=2, diversity=None,
                              history_genome_every=10).run()
    assert len(result.history) == 26
    for stats in result.history:
        assert (stats.best_genome is not None) == (stats.generation % 10 == 0)
        assert stats.best_fitness > 0  # statistics are still there
    assert len(result.best.genome) > 0  # the final best is untouched


def test_snapshots_write_png_csv_and_gif_from_disk(tmp_path):
    from PIL import Image

    from carmelolg.gapoly.callbacks import Snapshots

    problem = ImageProblem(gradient_target(), Limits(max_elements=20), initial_elements=5)
    snapshots = Snapshots(problem, tmp_path, every=5, output_width=64, frame_width=32)
    GeneticAlgorithm(problem, population_size=10, max_generations=12, seed=2, diversity=None,
                     callbacks=[snapshots]).run()
    pngs = sorted(p.name for p in (tmp_path / "snapshots").glob("gen_*.png"))
    assert pngs == ["gen_000000.png", "gen_000005.png", "gen_000010.png"]
    assert Image.open(tmp_path / "final.png").size == (64, 64)
    with Image.open(tmp_path / "evolution.gif") as gif:
        assert gif.n_frames == len(pngs) + 1 and gif.size == (32, 32)  # snapshots + final
    rows = (tmp_path / "history.csv").read_text().splitlines()
    assert rows[0] == "generation,best_mse,mean_mse,elements" and len(rows) == 14  # header + generations 0..12


# ---- every shape of element ---------------------------------------------------------------------------------------

SHAPE_PARAMS = pytest.mark.parametrize("shape", list(SHAPES.values()), ids=list(SHAPES))


def shape_problem(shape, **kwargs):
    return ImageProblem(gradient_target(), Limits(max_elements=20, min_size=0.02), initial_elements=5, shape=shape, **kwargs)


def shape_run(shape, seed, **kwargs):
    return GeneticAlgorithm(shape_problem(shape), population_size=16, max_generations=40, mutation_rate=0.1, seed=seed,
                            diversity=None, **kwargs).run()


@SHAPE_PARAMS
def test_every_shape_fitness_equals_numpy_mse_of_the_render(shape):
    problem = shape_problem(shape)
    genome = problem.random_genome(Random(3))
    assert genome.shape == (5, shape.cols) and genome.dtype == np.float32
    expected = float(np.mean((problem.render(genome) - problem.target) ** 2))
    assert problem.fitness(genome) == pytest.approx(expected, rel=1e-4)


@SHAPE_PARAMS
def test_every_shape_evolves_reproducibly_and_threads_match_serial(shape):
    first = shape_run(shape, 7)
    assert first.history[-1].best_fitness < first.history[0].best_fitness
    assert shape_run(shape, 7).best.fitness == first.best.fitness
    assert shape_run(shape, 7, workers=3, parallel="thread").best.fitness == first.best.fitness


@SHAPE_PARAMS
def test_every_shape_operators_keep_genomes_valid_and_inputs_untouched(shape):
    problem = shape_problem(shape)
    limits, rng = problem.limits, Random(1)
    a, b = problem.random_genome(rng), problem.random_genome(rng)
    a0, b0 = a.copy(), b.copy()
    operators.crossover(a, b, rng, limits)
    shape.mutate(a, rng, 1.0, limits, None)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)
    for _ in range(200):
        a = shape.mutate(a, rng, 0.3, limits, 0.5)
        a, b = operators.crossover(a, b, rng, limits)
        for genome in (a, b):
            assert genome.shape[1] == shape.cols and genome.dtype == np.float32
            assert limits.min_elements <= len(genome) <= limits.max_elements
            values = genome[:, 1:] if shape.name == "polygon" else genome  # a polygon row starts with its vertex count
            assert np.all((values >= 0) & (values <= 1))
    assert shape.describe(a).count("\n") == len(a) - 1


@SHAPE_PARAMS
def test_every_shape_renders_an_empty_genome_as_the_background(shape):
    image = render(shape.empty_genome(), 4, 3, (0.2, 0.4, 0.6), shape)
    assert image.shape == (3, 4, 3) and image[0, 0].tolist() == pytest.approx([0.2, 0.4, 0.6])


# ---- discs -------------------------------------------------------------------------------------------------------

def test_disc_covers_the_expected_pixels():
    big = np.stack([discs.make_row(0.5, 0.5, 0.25, (1.0, 0.0, 0.0, 1.0))])
    image = render(big, 40, 40, (0.0, 0.0, 0.0), DISC)
    covered = int((image[..., 0] > 0.5).sum())
    assert covered == pytest.approx(np.pi * 10**2, rel=0.08)  # radius 0.25 of 40 px = 10 px
    assert image[20, 20].tolist() == [1.0, 0.0, 0.0] and image[0, 0].tolist() == [0.0, 0.0, 0.0]


def test_a_disc_is_never_smaller_than_the_pixel_under_its_centre():
    tiny = np.stack([discs.make_row(0.52, 0.27, 0.0, (1.0, 1.0, 1.0, 1.0))])
    image = render(tiny, 10, 10, (0.0, 0.0, 0.0), DISC)
    assert int((image[..., 0] > 0).sum()) == 1 and image[2, 5, 0] == 1.0


def test_disc_radius_is_a_fraction_of_the_longest_side_so_discs_stay_round():
    wide = np.stack([discs.make_row(0.5, 0.5, 0.2, (1.0, 1.0, 1.0, 1.0))])
    image = render(wide, 100, 50, (0.0, 0.0, 0.0), DISC)
    ys, xs = np.nonzero(image[..., 0] > 0.5)
    assert xs.max() - xs.min() == pytest.approx(ys.max() - ys.min(), abs=2)  # 40 px wide, 40 px tall


def test_numba_discs_match_pillow():
    rng, limits = Random(5), Limits(max_size=0.4, min_size=0.02)
    genome = np.stack([discs.random_row(rng, limits) for _ in range(30)])
    ours = render(genome, 64, 64, (0.5, 0.5, 0.5), DISC)
    canvas = np.full((64, 64, 3), 0.5, dtype=np.float32)
    for row in genome:
        mask = Image.new("L", (64, 64), 0)
        cx, cy, r = row[discs.X] * 64, row[discs.Y] * 64, max(row[discs.RADIUS] * 64, 0.7072)
        ImageDraw.Draw(mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0 * row[discs.ALPHA]
        canvas = canvas * (1 - alpha) + row[discs.RED : discs.BLUE + 1] * alpha
    assert np.abs(ours - canvas).mean() < 0.01


def test_disc_operators_respect_the_size_limits():
    limits = Limits(max_elements=15, max_size=0.2, min_size=0.04, min_alpha=0.1, max_alpha=0.9)
    problem = ImageProblem(gradient_target(), limits, 8, shape=DISC)
    rng = Random(2)
    genome = problem.random_genome(rng)
    for _ in range(300):
        genome = DISC.mutate(genome, rng, 0.3, limits, 0.8)
        radius = genome[:, discs.RADIUS]
        assert np.all(radius >= limits.min_size / 2 - 1e-6) and np.all(radius <= limits.max_size / 2 + 1e-6)
        assert np.all((genome[:, discs.ALPHA] >= 0.1 - 1e-6) & (genome[:, discs.ALPHA] <= 0.9 + 1e-6))


def test_disc_row_layout_is_the_documented_one():
    """If this fails, update the diagram in discs.py and the numba kernel too."""
    row = discs.make_row(0.1, 0.2, 0.3, (0.4, 0.5, 0.6, 0.7))
    assert discs.COLS == 7 and (discs.X, discs.Y, discs.RADIUS, discs.RED, discs.ALPHA) == (0, 1, 2, 3, 6)
    assert row.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert discs.describe(np.stack([row])) == "#0 disc (0.10,0.20) radius 0.300 rgba(0.40,0.50,0.60,0.70)"


def test_command_line_runs_every_shape(tmp_path):
    from carmelolg.gapoly.cli import main

    photo = tmp_path / "photo.png"
    Image.fromarray((gradient_target(32) * 255).astype(np.uint8), "RGB").save(photo)
    for name in SHAPES:
        out = tmp_path / name
        assert main([str(photo), "--shape", name, "--generations", "6", "--population", "8", "--size", "24",
                     "--elements", "30", "--snapshot-every", "3", "--workers", "1", "--seed", "1", "--out", str(out)]) == 0
        assert (out / "final.png").exists() and (out / "evolution.gif").exists()
        assert len((out / "history.csv").read_text().splitlines()) == 8  # header + generations 0..6


# ---- Voronoi points ----------------------------------------------------------------------------------------------

def brute_force_voronoi(genome, width, height):
    """Colour of the nearest seed for every pixel, checking all seeds (no grid)."""
    ys, xs = np.mgrid[0:height, 0:width]
    px, py = xs + 0.5, ys + 0.5
    sx, sy = genome[:, voronoi.X].astype(np.float64) * width, genome[:, voronoi.Y].astype(np.float64) * height
    nearest = ((px[..., None] - sx) ** 2 + (py[..., None] - sy) ** 2).argmin(axis=-1)
    return genome[nearest][..., voronoi.RED : voronoi.BLUE + 1]


@pytest.mark.parametrize(("count", "width", "height"), [(1, 20, 20), (7, 33, 17), (300, 64, 40), (900, 50, 90)])
def test_voronoi_grid_search_is_exact(count, width, height):
    rng = np.random.default_rng(count)
    genome = rng.random((count, voronoi.COLS), dtype=np.float32)
    ours = render(genome, width, height, (0.0, 0.0, 0.0), VORONOI)
    assert np.array_equal(ours, brute_force_voronoi(genome, width, height))


def test_voronoi_grid_search_is_exact_with_clustered_seeds_and_seeds_on_the_border():
    rng = np.random.default_rng(3)
    genome = rng.random((200, voronoi.COLS), dtype=np.float32)
    genome[:150, voronoi.X : voronoi.Y + 1] = 0.9 + 0.1 * genome[:150, voronoi.X : voronoi.Y + 1]  # one crowded corner
    genome[150:160, voronoi.X] = 0.0
    genome[160:170, voronoi.Y] = 1.0
    ours = render(genome, 48, 36, (0.0, 0.0, 0.0), VORONOI)
    assert np.array_equal(ours, brute_force_voronoi(genome, 48, 36))


def test_voronoi_two_seeds_split_the_image_in_two():
    genome = np.stack([voronoi.make_row(0.25, 0.5, (1.0, 0.0, 0.0)), voronoi.make_row(0.75, 0.5, (0.0, 0.0, 1.0))])
    image = render(genome, 20, 10, (0.0, 0.0, 0.0), VORONOI)
    assert image[5, 2].tolist() == [1.0, 0.0, 0.0] and image[5, 17].tolist() == [0.0, 0.0, 1.0]
    assert (image[..., 0] > 0.5).sum() == 100  # the left half


def test_voronoi_order_of_the_seeds_does_not_matter():
    rng = np.random.default_rng(1)
    genome = rng.random((40, voronoi.COLS), dtype=np.float32)
    shuffled = genome[rng.permutation(40)]
    assert np.array_equal(render(genome, 30, 30, (0.0, 0.0, 0.0), VORONOI),
                          render(shuffled, 30, 30, (0.0, 0.0, 0.0), VORONOI))


def test_voronoi_row_layout_is_the_documented_one():
    """If this fails, update the diagram in voronoi.py and the numba kernel too."""
    row = voronoi.make_row(0.1, 0.2, (0.3, 0.4, 0.5))
    assert voronoi.COLS == 5 and (voronoi.X, voronoi.Y, voronoi.RED, voronoi.BLUE) == (0, 1, 2, 4)
    assert row.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert voronoi.describe(np.stack([row])) == "#0 seed (0.10,0.20) rgb(0.30,0.40,0.50)"


# ---- mosaic ------------------------------------------------------------------------------------------------------

def hexagon_area(radius):
    return 1.5 * np.sqrt(3) * radius**2


def test_mosaic_tile_is_an_opaque_regular_hexagon():
    radius = mosaic.quantise(0.2)  # tiles are drawn at the radius of their size level
    row = mosaic.make_row(0.5, 0.5, 0.2, (1.0, 0.0, 0.0))
    image = render(np.stack([row]), 100, 100, (0.0, 0.0, 0.0), MOSAIC)
    covered = image[..., 0] > 0.5
    assert int(covered.sum()) == pytest.approx(hexagon_area(radius * 100), rel=0.05)
    assert set(image[covered][:, 0]) == {1.0} and image[0, 0].tolist() == [0.0, 0.0, 0.0]


def test_mosaic_tile_has_a_flat_top_and_pointed_sides():
    row = mosaic.make_row(0.5, 0.5, 0.2, (1.0, 1.0, 1.0))
    image = render(np.stack([row]), 100, 100, (0.0, 0.0, 0.0), MOSAIC)[..., 0] > 0.5
    ys, xs = np.nonzero(image)
    assert xs.max() - xs.min() > ys.max() - ys.min()  # width 2r, height sqrt(3) r
    cy = (ys.min() + ys.max()) // 2
    assert image[cy, xs.min() + 1] and not image[ys.min(), xs.min() + 1]  # a vertex at the side, a flat edge on top


def test_mosaic_snaps_the_tile_to_the_lattice_so_small_moves_change_nothing():
    here = np.stack([mosaic.make_row(0.5, 0.5, 0.2, (1.0, 0.0, 0.0))])
    nudged = np.stack([mosaic.make_row(0.5 + 0.004, 0.5 - 0.004, 0.2, (1.0, 0.0, 0.0))])
    black = (0.0, 0.0, 0.0)
    assert np.array_equal(render(here, 100, 100, black, MOSAIC), render(nudged, 100, 100, black, MOSAIC))
    cx, cy = mosaic.snap(50.0, 50.0, 20.0)
    assert mosaic.snap(cx, cy, 20.0) == pytest.approx((cx, cy))  # a lattice node snaps to itself


def test_mosaic_tiles_of_one_size_cover_the_image_without_gaps():
    side = 90
    radius = mosaic.quantise(0.06) * side
    rows = []
    for i in range(int(side / (1.5 * radius)) + 3):
        for j in range(int(side / (np.sqrt(3) * radius)) + 3):
            cx, cy = 1.5 * radius * i, np.sqrt(3) * radius * (j + 0.5 * (i % 2))
            rows.append(mosaic.make_row(cx / side, cy / side, mosaic.quantise(0.06), (1.0, 1.0, 1.0)))
    image = render(np.stack(rows), side, side, (0.0, 0.0, 0.0), MOSAIC)[..., 0]
    assert (image > 0.5).mean() > 0.995  # the lattice tiles the plane (shared edges aside)


def test_mosaic_radius_is_rounded_to_size_levels():
    levels = {float(mosaic.quantise(r)) for r in np.linspace(0.01, 0.2, 400)}
    assert len(levels) < 12  # a few sizes, not a continuum
    ordered = sorted(levels)
    assert all(b / a == pytest.approx(np.sqrt(2), rel=1e-6) for a, b in itertools.pairwise(ordered))


def test_mosaic_operators_respect_the_size_limits():
    limits = Limits(max_elements=15, max_size=0.2, min_size=0.04)
    problem = ImageProblem(gradient_target(), limits, 8, shape=MOSAIC)
    rng = Random(2)
    genome = problem.random_genome(rng)
    for _ in range(300):
        genome = MOSAIC.mutate(genome, rng, 0.3, limits, 0.8)
        radius = genome[:, mosaic.RADIUS]
        assert np.all(radius >= limits.min_size / 2 - 1e-6) and np.all(radius <= limits.max_size / 2 + 1e-6)


def test_mosaic_row_layout_is_the_documented_one():
    """If this fails, update the diagram in mosaic.py and the numba kernel too."""
    row = mosaic.make_row(0.1, 0.2, 0.3, (0.4, 0.5, 0.6))
    assert mosaic.COLS == 6 and (mosaic.X, mosaic.Y, mosaic.RADIUS, mosaic.RED, mosaic.BLUE) == (0, 1, 2, 3, 5)
    assert row.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert mosaic.describe(np.stack([row])) == "#0 tile (0.10,0.20) radius 0.300 rgb(0.40,0.50,0.60)"


# ---- glyphs ------------------------------------------------------------------------------------------------------

def solid_masks(count=1):
    return np.ones((count, glyphs.ATLAS_SIZE, glyphs.ATLAS_SIZE), dtype=np.float32)


def render_glyphs(genome, width, height, background, masks):
    return glyphs.render_image(genome, width, height, np.asarray(background, dtype=np.float32), masks)


def test_glyph_atlas_has_ink_for_every_character():
    masks = glyphs.atlas()
    assert masks.shape == (len(glyphs.GLYPHS), glyphs.ATLAS_SIZE, glyphs.ATLAS_SIZE) and masks.dtype == np.float32
    assert np.all(masks.max(axis=(1, 2)) > 0.5)  # no blank character
    assert np.all((masks >= 0) & (masks <= 1))


def test_glyph_box_is_a_square_of_the_given_size_so_glyphs_keep_their_shape():
    row = glyphs.make_row(0.5, 0.5, 0.5, 0.0, 0.0, (1.0, 0.0, 0.0, 1.0))
    image = render_glyphs(np.stack([row]), 80, 40, (0.0, 0.0, 0.0), solid_masks())  # size 0.5 of the longest side = 40 px
    ys, xs = np.nonzero(image[..., 0] > 0.5)
    assert xs.max() - xs.min() + 1 == 40 and ys.max() - ys.min() + 1 == 40 - 0  # the box fills the image height
    assert image[20, 40].tolist() == [1.0, 0.0, 0.0] and image[0, 0].tolist() == [0.0, 0.0, 0.0]


def test_glyph_rotation_turns_the_character_around_its_centre():
    masks = solid_masks()
    masks[0, :, glyphs.ATLAS_SIZE // 2 :] = 0.0  # ink only in the left half of the box
    upright = np.stack([glyphs.make_row(0.5, 0.5, 0.5, 0.0, 0.0, (1.0, 1.0, 1.0, 1.0))])
    turned = np.stack([glyphs.make_row(0.5, 0.5, 0.5, 0.5, 0.0, (1.0, 1.0, 1.0, 1.0))])  # half a turn
    a = render_glyphs(upright, 40, 40, (0.0, 0.0, 0.0), masks)[..., 0]
    b = render_glyphs(turned, 40, 40, (0.0, 0.0, 0.0), masks)[..., 0]
    assert a[20, 12] > 0.9 and a[20, 28] < 0.1
    assert b[20, 12] < 0.1 and b[20, 28] > 0.9


def test_glyph_picks_the_character_from_the_glyph_column():
    assert glyphs.glyph_char(0.0) == glyphs.GLYPHS[0] and glyphs.glyph_char(1.0) == glyphs.GLYPHS[-1]
    two = np.zeros((2, glyphs.ATLAS_SIZE, glyphs.ATLAS_SIZE), dtype=np.float32)
    two[1] = 1.0  # character 0 is blank, character 1 is solid
    low = np.stack([glyphs.make_row(0.5, 0.5, 0.5, 0.0, 0.1, (1.0, 1.0, 1.0, 1.0))])
    high = np.stack([glyphs.make_row(0.5, 0.5, 0.5, 0.0, 0.9, (1.0, 1.0, 1.0, 1.0))])
    assert render_glyphs(low, 20, 20, (0.0, 0.0, 0.0), two).max() == 0.0
    assert render_glyphs(high, 20, 20, (0.0, 0.0, 0.0), two)[10, 10, 0] == pytest.approx(1.0)


def test_glyph_operators_respect_the_size_limits_and_keep_angle_in_a_turn():
    limits = Limits(max_elements=15, max_size=0.2, min_size=0.04, min_alpha=0.1, max_alpha=0.9)
    problem = ImageProblem(gradient_target(), limits, 8, shape=GLYPH)
    rng = Random(2)
    genome = problem.random_genome(rng)
    for _ in range(300):
        genome = GLYPH.mutate(genome, rng, 0.3, limits, 0.8)
        assert np.all((genome[:, glyphs.SIZE] >= limits.min_size - 1e-6) & (genome[:, glyphs.SIZE] <= limits.max_size + 1e-6))
        assert np.all((genome[:, glyphs.ANGLE] >= 0) & (genome[:, glyphs.ANGLE] <= 1))
        assert np.all((genome[:, glyphs.ALPHA] >= 0.1 - 1e-6) & (genome[:, glyphs.ALPHA] <= 0.9 + 1e-6))


def test_glyph_row_layout_is_the_documented_one():
    """If this fails, update the diagram in glyphs.py and the numba kernel too."""
    row = glyphs.make_row(0.1, 0.2, 0.3, 0.4, 0.0, (0.5, 0.6, 0.7, 0.8))
    assert glyphs.COLS == 9 and (glyphs.X, glyphs.Y, glyphs.SIZE, glyphs.ANGLE, glyphs.GLYPH) == (0, 1, 2, 3, 4)
    assert (glyphs.RED, glyphs.ALPHA) == (5, 8)
    assert row.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.0, 0.5, 0.6, 0.7, 0.8])
    assert glyphs.describe(np.stack([row])) == (
        f"#0 glyph {glyphs.GLYPHS[0]!r} (0.10,0.20) size 0.300 angle 0.40 rgba(0.50,0.60,0.70,0.80)")


# ---- brush strokes -----------------------------------------------------------------------------------------------

def brute_force_stroke_mask(start, end, half, width, height):
    """Pixels whose centre is within ``half`` pixels of the segment ``start``-``end`` (given in pixels)."""
    ys, xs = np.mgrid[0:height, 0:width]
    p = np.stack([xs + 0.5, ys + 0.5], axis=-1)
    a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * (b - a)), axis=-1) <= half


def test_a_straight_stroke_covers_the_pixels_near_its_segment():
    # the control point is the midpoint, so the Bezier curve is the straight segment from (8, 20) to (32, 20)
    row = strokes.make_row((0.2, 0.5), (0.5, 0.5), (0.8, 0.5), 0.1, (1.0, 0.0, 0.0, 1.0))  # brush 0.1 of 40 px = 4 px
    image = render(np.stack([row]), 40, 40, (0.0, 0.0, 0.0), STROKE)
    expected = brute_force_stroke_mask((8, 20), (32, 20), 2.0, 40, 40)
    assert np.array_equal(image[..., 0] > 0.5, expected)


def test_a_curved_stroke_bends_towards_its_control_point():
    row = strokes.make_row((0.2, 0.5), (0.5, 0.1), (0.8, 0.5), 0.05, (1.0, 1.0, 1.0, 1.0))
    image = render(np.stack([row]), 100, 100, (0.0, 0.0, 0.0), STROKE)[..., 0]
    middle = (0.5 + 2 * 0.1 + 0.5) / 4  # y of the curve at t = 1/2: (y0 + 2 * cy + y1) / 4 = 0.3
    assert middle == pytest.approx(0.3) and image[30, 50] > 0.9  # the curve passes through (0.5, 0.3)
    assert image[50, 50] < 0.1 and image[10, 50] < 0.1  # neither the chord nor the control point is painted


def test_overlapping_parts_of_a_stroke_do_not_double_the_opacity():
    loop = strokes.make_row((0.3, 0.5), (0.9, 0.5), (0.3, 0.52), 0.1, (1.0, 1.0, 1.0, 0.5))  # doubles back on itself
    image = render(np.stack([loop]), 40, 40, (0.0, 0.0, 0.0), STROKE)[..., 0]
    assert image.max() == pytest.approx(0.5)


def test_stroke_operators_respect_the_size_and_width_limits():
    limits = Limits(max_elements=15, max_size=0.2, min_size=0.02, min_alpha=0.1, max_alpha=0.9)
    problem = ImageProblem(gradient_target(), limits, 8, shape=STROKE)
    rng = Random(2)
    genome = problem.random_genome(rng)
    for _ in range(300):
        genome = STROKE.mutate(genome, rng, 0.3, limits, 0.8)
        assert all(strokes.extent(row) <= limits.max_size + 1e-5 for row in genome)
        assert np.all((genome[:, strokes.WIDTH] >= limits.min_size - 1e-6)
                      & (genome[:, strokes.WIDTH] <= limits.max_size / 3 + 1e-6))
        assert np.all((genome[:, strokes.ALPHA] >= 0.1 - 1e-6) & (genome[:, strokes.ALPHA] <= 0.9 + 1e-6))


def test_stroke_row_layout_is_the_documented_one():
    """If this fails, update the diagram in strokes.py and the numba kernel too."""
    row = strokes.make_row((0.1, 0.2), (0.3, 0.4), (0.5, 0.6), 0.7, (0.8, 0.9, 1.0, 0.5))
    assert strokes.COLS == 11 and (strokes.X0, strokes.CX, strokes.X1, strokes.WIDTH) == (0, 2, 4, 6)
    assert (strokes.RED, strokes.ALPHA) == (7, 10)
    assert row.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.5])
    assert strokes.describe(np.stack([row])) == (
        "#0 stroke (0.10,0.20) via (0.30,0.40) to (0.50,0.60) width 0.700 rgba(0.80,0.90,1.00,0.50)")


def test_shapes_are_strategies_registered_under_their_cli_name():
    assert set(SHAPES) == {"polygon", "disc", "voronoi", "mosaic", "glyph", "stroke"}
    for name, shape in SHAPES.items():
        assert isinstance(shape, Shape) and shape.name == name
    with pytest.raises(TypeError):
        Shape()  # the interface cannot be instantiated: each shape implements it


def test_removed_command_line_options_are_gone(tmp_path, capsys):
    from carmelolg.gapoly.cli import main

    for option in ("--polygons", "--points", "--initial-elements", "--parallel", "--history-every"):
        with pytest.raises(SystemExit):
            main([str(tmp_path / "photo.png"), option, "1"])
        assert "unrecognized arguments" in capsys.readouterr().err
