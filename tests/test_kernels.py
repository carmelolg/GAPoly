"""The Numba kernels against their own pure-Python source, plus the small pieces of the package no other test reaches.

Coverage cannot see inside compiled code, so each kernel is also run through ``py_func`` (the original Python function
Numba compiled) and must give the same pixels as the compiled one: this checks the compilation and lets the lines count.
"""
import runpy
import sys
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from carmelolg.gapoly import SHAPES, ImageProblem, Limits
from carmelolg.gapoly.cli.help_format import colorize
from carmelolg.gapoly.core import kernels
from carmelolg.gapoly.shapes import discs, glyphs, mosaic, polygons, strokes, voronoi

WIDTH, HEIGHT = 23, 17
BACKGROUND = np.array([0.1, 0.2, 0.3], dtype=np.float32)
KERNEL_MODULES = {"polygon": polygons, "disc": discs, "voronoi": voronoi, "mosaic": mosaic, "stroke": strokes, "glyph": glyphs}
LIMITS = Limits(max_elements=40, max_size=0.6, min_size=0.02)  # big elements: they cross the borders of the canvas


def genome_for(shape, count=14, seed=5):
    rng = Random(seed)
    genome = np.stack([shape.random_row(rng, LIMITS) for _ in range(count)]).astype(np.float32)
    genome[0, :2] = 0.0, 0.0  # an element right on the corner, one on the opposite corner, one in the middle
    genome[1, :2] = 1.0, 1.0
    genome[2, :2] = 0.5, 0.5
    return np.ascontiguousarray(genome)


def window_args(name):
    """The image-window arguments of ``_draw``: every shape but Voronoi draws a window of the image."""
    return (0, 0, WIDTH, HEIGHT) if SHAPES[name].incremental else ()


def extra_args(name):
    return (glyphs.atlas(),) if name == "glyph" else ()


@pytest.mark.parametrize("name", list(KERNEL_MODULES))
def test_python_source_of_the_kernels_gives_the_same_pixels_as_the_compiled_ones(name):
    module, shape = KERNEL_MODULES[name], SHAPES[name]
    genome, extra = genome_for(shape), extra_args(name)

    compiled = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    module.render_into(genome, BACKGROUND, compiled, *extra)
    plain = np.empty_like(compiled)
    module.render_into.py_func(genome, BACKGROUND, plain, *extra)
    assert np.allclose(plain, compiled, atol=1e-5)

    drawn = np.empty_like(compiled)
    kernels.fill(drawn, BACKGROUND)
    module._draw.py_func(genome, drawn, *window_args(name), *extra)
    assert np.allclose(drawn, compiled, atol=1e-5)

    target = np.random.default_rng(0).random((HEIGHT, WIDTH, 3), dtype=np.float32)
    expected = float(np.mean((compiled - target) ** 2))
    assert module.render_mse.py_func(genome, target, BACKGROUND, *extra) == pytest.approx(expected, rel=1e-4)
    assert module.render_mse(genome, target, BACKGROUND, *extra) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("name", list(KERNEL_MODULES))
def test_kernels_draw_nothing_for_an_empty_genome(name):
    module, shape = KERNEL_MODULES[name], SHAPES[name]
    canvas = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    module.render_into.py_func(shape.empty_genome(), BACKGROUND, canvas, *extra_args(name))
    assert np.allclose(canvas, BACKGROUND)
    canvas[:] = BACKGROUND
    module._draw.py_func(shape.empty_genome(), canvas, *window_args(name), *extra_args(name))
    assert np.allclose(canvas, BACKGROUND)


def test_shared_kernels_fill_and_measure():
    canvas = np.zeros((4, 5, 3), dtype=np.float32)
    kernels.fill.py_func(canvas, BACKGROUND)
    assert np.allclose(canvas, BACKGROUND)
    target = np.zeros_like(canvas)
    assert kernels.canvas_mse.py_func(canvas, target) == pytest.approx(float(np.mean(BACKGROUND**2)), rel=1e-5)
    assert kernels.canvas_mse.py_func(canvas, canvas) == 0.0


@pytest.mark.parametrize("radius", [0.004, 0.01, 0.03, 0.08, 0.2])
def test_mosaic_level_and_lattice_helpers_match_their_compiled_versions(radius):
    assert mosaic.quantise.py_func(radius) == pytest.approx(mosaic.quantise(radius))
    plain, compiled = mosaic.snap.py_func(0.37, 0.61, radius), mosaic.snap(0.37, 0.61, radius)
    assert plain == pytest.approx(compiled)


def test_glyph_sampling_is_zero_outside_the_mask_and_matches_when_compiled():
    mask = glyphs.atlas()[0]
    for u, v in [(0.5, 0.5), (0.2, 0.8), (0.0, 0.0), (1.0, 1.0), (-0.3, 0.5), (0.5, 1.7)]:
        assert glyphs._sample.py_func(mask, u, v) == pytest.approx(glyphs._sample(mask, u, v))


# ---- operators, help, entry point, live window --------------------------------------------------------------------

@pytest.mark.parametrize("pick", [kernels.pick, kernels.pick.py_func])
def test_pick_chooses_nothing_at_probability_zero_and_everything_at_one(pick):
    out = np.empty(5, dtype=np.int64)
    assert pick(5, 0.0, out) == 0
    assert pick(5, 1.0, out) == 5 and list(out) == [0, 1, 2, 3, 4]
    np.random.seed(0)
    assert 0 < pick(5, 0.5, out) <= 5 or pick(5, 0.5, out) == 0  # in range either way


def test_colorize_leaves_text_that_is_already_coloured():
    assert colorize("\033[1mAlready\033[0m") == "\033[1mAlready\033[0m"


def test_python_dash_m_runs_the_command_line(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gapoly", "--help"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("carmelolg.gapoly", run_name="__main__")
    assert stop.value.code == 0
    assert "--shape" in capsys.readouterr().out


@pytest.fixture
def agg(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
    yield shown
    plt.close("all")


@pytest.mark.parametrize(("width", "height"), [(30, 20), (20, 30)], ids=["landscape", "portrait"])
def test_live_view_updates_the_window_and_shows_it_at_the_end(agg, width, height):
    from carmelolg.gapoly.callbacks.live import LiveView

    shape = SHAPES["polygon"]
    target = np.random.default_rng(1).random((height, width, 3), dtype=np.float32)
    problem = ImageProblem(target, Limits(max_elements=10), initial_elements=3, shape=shape)
    live = LiveView(problem, every=2)
    genome = problem.random_genome(Random(1))
    for generation in range(4):
        live.on_generation(SimpleNamespace(generation=generation, best_fitness=1.0 / (generation + 1), best_genome=genome), None)
    assert live.x == [0, 1, 2, 3] and len(live.y) == 4
    assert "generation 2" in live.a_best.get_title()  # updated on the generations that are multiples of `every`
    live.on_end(None, None)
    assert agg == [True]


def test_command_line_with_the_live_window(agg, tmp_path):
    from carmelolg.gapoly.cli import main

    photo = tmp_path / "photo.png"
    Image.fromarray(np.random.default_rng(2).integers(0, 255, (20, 28, 3), dtype=np.uint8), "RGB").save(photo)
    assert main([str(photo), "--live", "--live-every", "2", "--generations", "4", "--population", "6", "--size", "20",
                 "--elements", "10", "--workers", "1", "--seed", "1", "--out", str(tmp_path / "out")]) == 0
    assert agg == [True]


def box_helper(name, row):
    """The pixel box of one row, from the Python source of the shape's box helper."""
    module = KERNEL_MODULES[name]
    if name == "stroke":
        return module._curve.py_func(row, WIDTH, HEIGHT, np.empty(33), np.empty(33))[2:]
    if name == "mosaic":
        return module._tile.py_func(row, WIDTH, HEIGHT)[3:]
    return module._box.py_func(row, WIDTH, HEIGHT)


@pytest.mark.parametrize("name", [n for n, s in SHAPES.items() if s.incremental])
def test_window_kernels_python_source_gives_the_same_numbers_as_the_compiled_ones(name):
    module, shape = KERNEL_MODULES[name], SHAPES[name]
    genome, extra = genome_for(shape), extra_args(name)
    if name == "polygon":
        genome[3, polygons.N_VERTICES] = 2  # fewer than 3 vertices: nothing is drawn, the box is empty
    target = np.random.default_rng(1).random((HEIGHT, WIDTH, 3), dtype=np.float32)
    for x0, y0, x1, y1 in [(0, 0, WIDTH, HEIGHT), (4, 3, 17, 12), (WIDTH - 3, HEIGHT - 2, WIDTH, HEIGHT), (0, 0, 1, 1)]:
        compiled = module.window_sse(genome, target, BACKGROUND, x0, y0, x1, y1, *extra)
        assert module.window_sse.py_func(genome, target, BACKGROUND, x0, y0, x1, y1, *extra) == pytest.approx(compiled, rel=1e-4)
        window = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
        kernels.fill(window, BACKGROUND)
        module._draw.py_func(genome, window, x0, y0, WIDTH, HEIGHT, *extra)  # the Python source of the windowed drawing
        assert kernels.window_error(window, target, x0, y0) == pytest.approx(compiled, rel=1e-4)
    boxes = module.boxes(genome, WIDTH, HEIGHT)
    assert (module.boxes.py_func(genome, WIDTH, HEIGHT) == boxes).all()
    assert [tuple(box_helper(name, row)) for row in genome] == [tuple(box) for box in boxes]


def test_shared_row_comparison_and_boxes_python_source():
    a = np.array([1, 2, 3], dtype=np.float32)
    assert kernels._same_row.py_func(a, a.copy())
    assert not kernels._same_row.py_func(a, a + np.array([0, 0, 1], dtype=np.float32))


@pytest.mark.parametrize("name", [n for n, s in SHAPES.items() if s.incremental])
def test_delta_kernel_is_the_difference_of_the_two_full_errors(name):
    module, shape = KERNEL_MODULES[name], SHAPES[name]
    before, extra = genome_for(shape), extra_args(name)
    target = np.random.default_rng(2).random((HEIGHT, WIDTH, 3), dtype=np.float32)
    rng = Random(8)
    edited = before.copy()
    edited[5] = shape.random_row(rng, LIMITS)
    changes = [edited, np.insert(before, 7, shape.random_row(rng, LIMITS), axis=0), np.delete(before, 3, axis=0)]
    for after in changes:
        expected = (float(((shape.render_image(after, WIDTH, HEIGHT, BACKGROUND) - target) ** 2).sum())
                    - float(((shape.render_image(before, WIDTH, HEIGHT, BACKGROUND) - target) ** 2).sum()))
        everything = WIDTH * HEIGHT
        for function in (module.delta_sse, module.delta_sse.py_func):
            assert function(before, after, target, BACKGROUND, 64, everything, *extra) == pytest.approx(expected, rel=1e-3, abs=1e-4)
    assert module.delta_sse.py_func(before, before.copy(), target, BACKGROUND, 64, everything, *extra) == 0.0
    for function in (module.delta_sse, module.delta_sse.py_func):
        assert np.isnan(function(before, np.flip(before, axis=0).copy(), target, BACKGROUND, 2, everything, *extra))  # too many rows
        assert np.isnan(function(before, edited, target, BACKGROUND, 64, 0, *extra))  # too many pixels


# ---- compiled mutation -------------------------------------------------------------------------------------------

EDITORS = {  # the (row, limits) functions of each shape's compiled mutation, by module attribute
    "polygon": ["_move_vertex", "_shift_polygon", "_edit_shape", "_change_color"],
    "stroke": ["_move_point", "_shift_stroke", "_resize", "_edit_shape", "_change_color"],
    "disc": ["_move", "_resize", "_edit_shape", "_change_color"],
    "glyph": ["_move", "_resize", "_rotate", "_edit_shape", "_change_color"],
    "mosaic": ["_move", "_resize", "_edit_shape", "_change_color"],
    "voronoi": ["_edit_shape", "_change_color"],
}
BOUNDS = {"stroke": "_width_bounds", "disc": "_radius_bounds", "glyph": "_size_bounds", "mosaic": "_radius_bounds"}
LIM = kernels.limits_array(LIMITS)


def valid_genome(shape, count=12):
    """Rows as the program makes them (``genome_for`` pokes the first columns, which for polygons are the vertex count)."""
    return np.ascontiguousarray(np.stack([shape.random_row(Random(7 + k), LIMITS) for k in range(count)]))


def test_limits_array_follows_the_l_constants_and_is_read_only():
    assert LIM[kernels.L_MAX_ELEMENTS] == LIMITS.max_elements and LIM[kernels.L_MAX_SIZE] == LIMITS.max_size
    assert LIM[kernels.L_MIN_ALPHA] == LIMITS.min_alpha and LIM[kernels.L_MIN_VERTICES] == LIMITS.min_vertices
    assert kernels.limits_array(LIMITS) is LIM  # cached
    with pytest.raises(ValueError):
        LIM[0] = 5


@pytest.mark.parametrize("name", list(EDITORS))
def test_python_source_of_the_row_editors_keeps_rows_valid(name):
    module, shape = KERNEL_MODULES[name], SHAPES[name]
    np.random.seed(3)
    for editor in EDITORS[name]:
        for _ in range(60):  # enough draws to take every branch of the random choices
            row = valid_genome(shape)[3].copy()
            getattr(module, editor).py_func(row, LIM)
            values = row[1:] if name == "polygon" else row  # the first column of a polygon is its vertex count
            assert np.isfinite(row).all() and values.min() >= 0.0 and values.max() <= 1.0 + 1e-6
    if name in BOUNDS:
        low, high = getattr(module, BOUNDS[name]).py_func(LIM)
        assert 0 < low <= high
    if name in ("polygon", "stroke"):
        assert module._extent.py_func(valid_genome(shape)[3]) >= 0.0


@pytest.mark.parametrize("name", list(EDITORS))
def test_random_rows_honour_the_centre_and_colour_asked_for_and_are_random_otherwise(name):
    module, shape = KERNEL_MODULES[name], SHAPES[name]
    row = shape.random_row(Random(1), LIMITS, (0.25, 0.5, 0.75), (0.4, 0.6))
    assert row.dtype == np.float32 and len(row) == shape.cols
    assert np.allclose(row[[module.RED, module.GREEN, module.BLUE]], (0.25, 0.5, 0.75))
    if name not in ("polygon", "stroke"):  # the other rows hold their centre in X and Y
        assert np.allclose(row[[module.X, module.Y]], (0.4, 0.6))
    a, b = shape.random_row(Random(1), LIMITS), shape.random_row(Random(2), LIMITS)
    assert (a == shape.random_row(Random(1), LIMITS)).all() and not (a == b).all()
    np.random.seed(0)
    plain = module._random_row.py_func(LIM, -1.0, -1.0, -1.0, -1.0, -1.0)
    assert np.isfinite(plain).all() and module._new_row.py_func(LIM).shape == plain.shape
    assert module._random_row_seeded.py_func(5, LIM, 0.2, 0.3, 0.1, 0.2, 0.3).shape == plain.shape


@pytest.mark.parametrize("name", list(EDITORS))
def test_python_source_of_the_whole_mutation_covers_every_operation(name):
    module, shape = KERNEL_MODULES[name], SHAPES[name]
    genome = valid_genome(shape)
    sizes = set()
    for seed in range(40):
        child = module._mutate.py_func(genome, seed, 1.0, 0.5, LIM)  # rate 1: add, remove and reorder every time
        assert child.shape[1] == shape.cols and child.dtype == np.float32
        sizes.add(len(child))
        compiled = module._mutate(genome, seed, 1.0, 0.5, LIM)
        assert compiled.shape[1] == shape.cols
    assert genome.shape == (12, shape.cols) and sizes <= {11, 12, 13}  # the input is untouched, one row in, one out
    full = np.ascontiguousarray(np.tile(genome, (6, 1))[: int(LIM[kernels.L_MAX_ELEMENTS])])  # at the cap: no more rows
    assert len(module._mutate.py_func(full, 1, 1.0, 0.0, LIM)) <= len(full)
    one = np.ascontiguousarray(genome[:1])  # at the minimum of one element: nothing removed, nothing to reorder
    assert len(module._mutate.py_func(one, 1, 0.0, 0.0, LIM)) == 1


def test_a_mutation_depends_only_on_its_seed_and_arguments_whatever_the_thread():
    from concurrent.futures import ThreadPoolExecutor

    genome = valid_genome(SHAPES["polygon"])
    expected = polygons._mutate(genome, 9, 0.4, 0.3, LIM)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: polygons._mutate(genome, 9, 0.4, 0.3, LIM), range(16)))
    assert all((r == expected).all() for r in results)
    assert not (polygons._mutate(genome, 10, 0.4, 0.3, LIM).shape == expected.shape and
                (polygons._mutate(genome, 10, 0.4, 0.3, LIM) == expected).all())


@pytest.mark.parametrize("function", ["insert_row", "delete_row", "drop_and_move", "clamp"])
def test_python_source_of_the_row_kernels(function):
    fn = getattr(kernels, function).py_func
    genome = np.arange(12, dtype=np.float32).reshape(4, 3)
    if function == "insert_row":
        out = fn(genome, 1, np.full(3, 99, dtype=np.float32))
        assert out.shape == (5, 3) and (out[1] == 99).all() and (out[2] == genome[1]).all()
    elif function == "delete_row":
        assert (fn(genome, 1) == np.delete(genome, 1, axis=0)).all()
    elif function == "drop_and_move":
        np.random.seed(2)
        for reorder in (True, False):
            out = fn(genome, 1.0, LIM, reorder)
            assert len(out) == 3 and all(tuple(row) in set(map(tuple, genome.tolist())) for row in out)  # one went away
        assert fn(genome[:1], 1.0, LIM, True).shape == (1, 3)  # one element: kept
    else:
        assert fn(-1.0, 0.0, 1.0) == 0.0 and fn(2.0, 0.0, 1.0) == 1.0 and fn(0.4, 0.0, 1.0) == 0.4


@pytest.mark.parametrize("name", ["polygon", "stroke"])
def test_moving_a_point_never_grows_the_element_past_the_size_cap(name):
    module, shape = KERNEL_MODULES[name], SHAPES[name]
    row = valid_genome(shape)[3]
    capped = kernels.limits_array(Limits(max_elements=40, max_size=float(module._extent.py_func(row)), min_size=0.02))
    mover = module._move_vertex if name == "polygon" else module._move_point
    np.random.seed(1)
    undone = 0
    for _ in range(200):
        moved = row.copy()
        mover.py_func(moved, capped)
        assert module._extent.py_func(moved) <= capped[kernels.L_MAX_SIZE] + 1e-6
        undone += bool((moved == row).all())
    assert undone > 0  # some moves would have grown it: those were taken back


def test_a_stroke_of_zero_length_is_still_a_valid_row():
    np.random.seed(0)
    row = strokes._random_row.py_func(kernels.limits_array(Limits(max_size=0.0, min_size=0.0)), -1.0, -1.0, -1.0, -1.0, -1.0)
    assert np.isfinite(row).all() and row[strokes.WIDTH] > 0
