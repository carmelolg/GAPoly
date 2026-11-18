import re
from random import Random

import numpy as np
import pytest
from carmelolg.galib import GeneticAlgorithm
from PIL import Image, ImageDraw

from carmelolg.gapoly import KINDS, ImageProblem, Limits, PolygonProblem, discs, operators
from carmelolg.gapoly.kinds import DISC
from carmelolg.gapoly.render import render
from carmelolg.gapoly.shapes import (
    ALPHA,
    COLS,
    N_VERTICES,
    RED,
    X,
    Y,
    color,
    describe,
    empty_genome,
    extent,
    make_row,
    random_row,
    vertices,
)

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


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
    assert render(empty_genome(), 4, 4, (0.2, 0.4, 0.6))[0, 0].tolist() == pytest.approx([0.2, 0.4, 0.6])


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
    problem = PolygonProblem(np.zeros((8, 8, 3), dtype=np.float32))
    assert problem.fitness(empty_genome()) == 0.0
    far = np.stack([make_row(SQUARE, (1.0, 1.0, 1.0, 0.9))])
    near = np.stack([make_row(SQUARE, (1.0, 1.0, 1.0, 0.1))])
    assert problem.fitness(near) < problem.fitness(far)


def test_fitness_equals_numpy_mse_of_the_render():
    problem = PolygonProblem(gradient_target(), Limits(max_size=0.5), initial_elements=10)
    genome = problem.random_genome(Random(3))
    expected = float(np.mean((problem.render(genome) - problem.target) ** 2))
    assert problem.fitness(genome) == pytest.approx(expected, rel=1e-4)


def test_operators_keep_genome_valid_and_inputs_untouched():
    limits = Limits(max_elements=12)
    problem = PolygonProblem(gradient_target(), limits, initial_elements=6)
    rng = Random(1)
    a, b = problem.random_genome(rng), problem.random_genome(rng)
    a0, b0 = a.copy(), b.copy()
    operators.mutate(a, rng, 1.0, limits)
    operators.crossover(a, b, rng, limits)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)
    for _ in range(300):
        a = operators.mutate(a, rng, 0.3, limits)
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
    problem = PolygonProblem(gradient_target(), Limits(max_elements=20), initial_elements=5)
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
    """If this fails, someone changed the columns: update the diagram in shapes.py and the numba kernel too."""
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
    problem = PolygonProblem(gradient_target(), Limits(max_elements=20), initial_elements=5)
    result = GeneticAlgorithm(problem, population_size=10, max_generations=25, seed=2, diversity=None,
                              history_genome_every=10).run()
    assert len(result.history) == 26
    for stats in result.history:
        assert (stats.best_genome is not None) == (stats.generation % 10 == 0)
        assert stats.best_fitness > 0  # statistics are still there
    assert len(result.best.genome) > 0  # the final best is untouched


def test_snapshots_write_png_csv_and_gif_from_disk(tmp_path):
    from PIL import Image

    from carmelolg.gapoly.viewer import Snapshots

    problem = PolygonProblem(gradient_target(), Limits(max_elements=20), initial_elements=5)
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


# ---- every kind of element ---------------------------------------------------------------------------------------

KIND_PARAMS = pytest.mark.parametrize("kind", list(KINDS.values()), ids=list(KINDS))


def kind_problem(kind, **kwargs):
    return ImageProblem(gradient_target(), Limits(max_elements=20, min_size=0.02), initial_elements=5, kind=kind, **kwargs)


def kind_run(kind, seed, **kwargs):
    return GeneticAlgorithm(kind_problem(kind), population_size=16, max_generations=40, mutation_rate=0.1, seed=seed,
                            diversity=None, **kwargs).run()


@KIND_PARAMS
def test_every_kind_fitness_equals_numpy_mse_of_the_render(kind):
    problem = kind_problem(kind)
    genome = problem.random_genome(Random(3))
    assert genome.shape == (5, kind.cols) and genome.dtype == np.float32
    expected = float(np.mean((problem.render(genome) - problem.target) ** 2))
    assert problem.fitness(genome) == pytest.approx(expected, rel=1e-4)


@KIND_PARAMS
def test_every_kind_evolves_reproducibly_and_threads_match_serial(kind):
    first = kind_run(kind, 7)
    assert first.history[-1].best_fitness < first.history[0].best_fitness
    assert kind_run(kind, 7).best.fitness == first.best.fitness
    assert kind_run(kind, 7, workers=3, parallel="thread").best.fitness == first.best.fitness


@KIND_PARAMS
def test_every_kind_operators_keep_genomes_valid_and_inputs_untouched(kind):
    problem = kind_problem(kind)
    limits, rng = problem.limits, Random(1)
    a, b = problem.random_genome(rng), problem.random_genome(rng)
    a0, b0 = a.copy(), b.copy()
    operators.crossover(a, b, rng, limits)
    kind.mutate(a, rng, 1.0, limits, None)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)
    for _ in range(200):
        a = kind.mutate(a, rng, 0.3, limits, 0.5)
        a, b = operators.crossover(a, b, rng, limits)
        for genome in (a, b):
            assert genome.shape[1] == kind.cols and genome.dtype == np.float32
            assert limits.min_elements <= len(genome) <= limits.max_elements
            values = genome[:, 1:] if kind.name == "polygon" else genome  # a polygon row starts with its vertex count
            assert np.all((values >= 0) & (values <= 1))
    assert kind.describe(a).count("\n") == len(a) - 1


@KIND_PARAMS
def test_every_kind_renders_an_empty_genome_as_the_background(kind):
    image = render(kind.empty_genome(), 4, 3, (0.2, 0.4, 0.6), kind)
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
    problem = ImageProblem(gradient_target(), limits, 8, kind=DISC)
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


def test_command_line_runs_every_kind(tmp_path):
    from carmelolg.gapoly.cli import main

    photo = tmp_path / "photo.png"
    Image.fromarray((gradient_target(32) * 255).astype(np.uint8), "RGB").save(photo)
    for name in KINDS:
        out = tmp_path / name
        assert main([str(photo), "--shape", name, "--generations", "6", "--population", "8", "--size", "24",
                     "--elements", "30", "--snapshot-every", "3", "--workers", "1", "--seed", "1", "--out", str(out)]) == 0
        assert (out / "final.png").exists() and (out / "evolution.gif").exists()
        assert len((out / "history.csv").read_text().splitlines()) == 8  # header + generations 0..6
