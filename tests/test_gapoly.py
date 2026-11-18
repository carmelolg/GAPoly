import re
from random import Random

import numpy as np
import pytest
from carmelolg.galib import GeneticAlgorithm
from PIL import Image, ImageDraw

from carmelolg.gapoly import Limits, PolygonProblem, operators
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
    problem = PolygonProblem(gradient_target(), Limits(max_size=0.5), initial_polygons=10)
    genome = problem.random_genome(Random(3))
    expected = float(np.mean((problem.render(genome) - problem.target) ** 2))
    assert problem.fitness(genome) == pytest.approx(expected, rel=1e-4)


def test_operators_keep_genome_valid_and_inputs_untouched():
    limits = Limits(max_polygons=12)
    problem = PolygonProblem(gradient_target(), limits, initial_polygons=6)
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
            assert limits.min_polygons <= len(genome) <= limits.max_polygons
            for r in genome:
                xs, ys = vertices(r)
                assert limits.min_vertices <= len(xs) <= limits.max_vertices
                assert np.all((xs >= 0) & (xs <= 1) & (ys >= 0) & (ys <= 1))
                assert np.all((color(r) >= 0) & (color(r) <= 1))
                assert limits.min_alpha <= r[ALPHA] <= limits.max_alpha
                assert extent(r) <= limits.max_size + 1e-6


def run(seed: int, **kwargs):
    problem = PolygonProblem(gradient_target(), Limits(max_polygons=20), initial_polygons=5)
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
    problem = PolygonProblem(gradient_target(), Limits(max_polygons=20), initial_polygons=5)
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

    problem = PolygonProblem(gradient_target(), Limits(max_polygons=20), initial_polygons=5)
    snapshots = Snapshots(problem, tmp_path, every=5, output_width=64, frame_width=32)
    GeneticAlgorithm(problem, population_size=10, max_generations=12, seed=2, diversity=None,
                     callbacks=[snapshots]).run()
    pngs = sorted(p.name for p in (tmp_path / "snapshots").glob("gen_*.png"))
    assert pngs == ["gen_000000.png", "gen_000005.png", "gen_000010.png"]
    assert Image.open(tmp_path / "final.png").size == (64, 64)
    with Image.open(tmp_path / "evolution.gif") as gif:
        assert gif.n_frames == len(pngs) + 1 and gif.size == (32, 32)  # snapshots + final
    rows = (tmp_path / "history.csv").read_text().splitlines()
    assert rows[0] == "generation,best_mse,mean_mse,polygons" and len(rows) == 14  # header + generations 0..12
