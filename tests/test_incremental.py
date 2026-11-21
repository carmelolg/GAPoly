"""Incremental fitness: windows, boxes, the parent / child diff and ``ImageProblem.fitness_from_parent``.

The invariant behind it all: elements the child shares with its parent (same rows, same order) paint the same
pixels, so the error differs only inside the boxes of the rows that changed, and drawing a window gives exactly the
pixels of the full image there.
"""
from random import Random

import numpy as np
import pytest
from carmelolg.galib import GeneticAlgorithm, Individual

from carmelolg.gapoly import SHAPES, ImageProblem, Limits
from carmelolg.gapoly import problem as problem_module
from carmelolg.gapoly.core import kernels

WIDTH, HEIGHT = 31, 23
BACKGROUND = np.array([0.1, 0.2, 0.3], dtype=np.float32)
LIMITS = Limits(max_elements=60, max_size=0.4, min_size=0.03)
INCREMENTAL = [name for name, shape in SHAPES.items() if shape.incremental]


def target_image(seed=0):
    return np.random.default_rng(seed).random((HEIGHT, WIDTH, 3), dtype=np.float32)


def genome_for(shape, count=20, seed=5):
    rng = Random(seed)
    genome = np.stack([shape.random_row(rng, LIMITS) for _ in range(count)]).astype(np.float32)
    genome[0, :2] = 0.0, 0.0  # elements right on the corners, so boxes get clipped
    genome[1, :2] = 1.0, 1.0
    return np.ascontiguousarray(genome)


def image_problem(name, seed=1, initial_elements=25, **kwargs):
    return ImageProblem(target_image(seed), LIMITS, initial_elements, 3.0, SHAPES[name], **kwargs)


def parent_of(problem, genome):
    fitness = problem.fitness(genome)
    return Individual(genome, fitness, -fitness)


def test_only_voronoi_is_not_incremental():
    assert [name for name in SHAPES if not SHAPES[name].incremental] == ["voronoi"]
    with pytest.raises(NotImplementedError):
        SHAPES["voronoi"].boxes(SHAPES["voronoi"].empty_genome(), WIDTH, HEIGHT)
    with pytest.raises(NotImplementedError):
        SHAPES["voronoi"].window_sse(SHAPES["voronoi"].empty_genome(), target_image(), BACKGROUND, 0, 0, 2, 2)


@pytest.mark.parametrize("name", INCREMENTAL)
def test_a_window_is_the_same_pixels_as_the_full_image(name):
    shape, target = SHAPES[name], target_image()
    genome = genome_for(shape)
    full = shape.render_image(genome, WIDTH, HEIGHT, BACKGROUND)
    assert shape.window_sse(genome, target, BACKGROUND, 0, 0, WIDTH, HEIGHT) == pytest.approx(
        float(((full - target) ** 2).sum()), rel=1e-5)
    for x0, y0, x1, y1 in [(3, 2, 20, 15), (0, 0, 5, 5), (WIDTH - 4, HEIGHT - 3, WIDTH, HEIGHT), (10, 10, 11, 11)]:
        crop = (full[y0:y1, x0:x1] - target[y0:y1, x0:x1]) ** 2
        assert shape.window_sse(genome, target, BACKGROUND, x0, y0, x1, y1) == pytest.approx(float(crop.sum()), rel=1e-5)


@pytest.mark.parametrize("name", INCREMENTAL)
def test_removing_a_row_changes_only_pixels_inside_its_box(name):
    shape = SHAPES[name]
    genome = genome_for(shape)
    boxes = shape.boxes(genome, WIDTH, HEIGHT)
    assert boxes.shape == (len(genome), 4) and boxes.dtype == np.int64
    full = shape.render_image(genome, WIDTH, HEIGHT, BACKGROUND)
    for k in range(len(genome)):
        without = shape.render_image(np.delete(genome, k, axis=0), WIDTH, HEIGHT, BACKGROUND)
        changed = np.argwhere((full != without).any(axis=2))
        x0, x1, y0, y1 = boxes[k]
        assert all(x0 <= x < x1 and y0 <= y < y1 for y, x in changed), f"row {k} paints outside its box"


@pytest.mark.parametrize("name", INCREMENTAL)
def test_fitness_from_parent_follows_a_long_chain_of_mutations(name):
    problem, rng = image_problem(name), Random(7)
    individual = parent_of(problem, problem.random_genome(rng))
    for _ in range(400):
        child = problem.mutate(individual.genome, rng, 0.3)
        value = problem.fitness_from_parent(child, individual)
        assert value == pytest.approx(problem.fitness(child), rel=1e-6)
        individual = Individual(child, value, -value)


@pytest.mark.parametrize("name", INCREMENTAL)
def test_each_kind_of_change_is_handled(name):
    problem, rng = image_problem(name), Random(2)
    genome = problem.random_genome(rng)
    parent = parent_of(problem, genome)
    row = problem.shape.random_row(rng, LIMITS)
    edited = genome.copy()
    edited[3] = problem.shape.random_row(rng, LIMITS)
    moved = np.insert(np.delete(genome, 2, axis=0), 9, genome[2], axis=0)
    swapped = genome.copy()
    swapped[[4, 5]] = swapped[[5, 4]]
    changes = [np.insert(genome, 0, row, axis=0), np.insert(genome, len(genome), row, axis=0), np.delete(genome, 0, axis=0),
               np.delete(genome, len(genome) - 1, axis=0), edited, moved, swapped, problem.shape.empty_genome()]
    for child in changes:
        assert problem.fitness_from_parent(child, parent) == pytest.approx(problem.fitness(child), rel=1e-6)


@pytest.mark.parametrize("name", INCREMENTAL)
def test_a_child_equal_to_its_parent_costs_nothing(name):
    problem = image_problem(name)
    parent = Individual(problem.random_genome(Random(1)), 123.0, -123.0)  # a fitness the render could not give
    assert problem.fitness_from_parent(parent.genome.copy(), parent) == 123.0


def test_voronoi_children_are_drawn_in_full():
    problem = image_problem("voronoi")
    rng = Random(3)
    parent = parent_of(problem, problem.random_genome(rng))
    child = problem.mutate(parent.genome, rng, 0.5)
    assert problem.fitness_from_parent(child, parent) == problem.fitness(child)


@pytest.mark.parametrize("name", INCREMENTAL)
def test_children_far_from_their_parent_are_drawn_in_full(name, monkeypatch):
    problem = image_problem(name)
    parent = parent_of(problem, problem.random_genome(Random(1)))
    other = problem.random_genome(Random(99))
    monkeypatch.setattr(problem_module, "MAX_CHANGED_ROWS", 4)  # too many rows differ
    assert problem.fitness_from_parent(other, parent) == problem.fitness(other)
    monkeypatch.setattr(problem_module, "MAX_CHANGED_ROWS", 64)
    monkeypatch.setattr(problem_module, "MAX_REDRAWN", 0.0)  # the windows are too big
    child = np.delete(parent.genome, 0, axis=0)
    assert problem.fitness_from_parent(child, parent) == problem.fitness(child)


def rows(*values):
    return np.array([[v, v + 0.5] for v in values], dtype=np.float32).reshape(-1, 2)



@pytest.mark.parametrize("function", [kernels.unmatched_rows, kernels.unmatched_rows.py_func])
def test_unmatched_rows_pairs_identical_rows_in_order(function):
    def run(parent, child, limit=8):
        old, new, n_old, n_new = function(parent, child, limit)
        return list(old[: max(n_old, 0)]), list(new[: max(n_new, 0)])

    assert run(rows(1, 2, 3), rows(1, 2, 3)) == ([], [])
    assert run(rows(1, 2, 3), rows(1, 9, 3)) == ([1], [1])  # edited
    assert run(rows(1, 2, 3), rows(1, 9, 2, 3)) == ([], [1])  # inserted
    assert run(rows(1, 2, 3), rows(2, 3)) == ([0], [])  # removed
    assert run(rows(1, 2, 3), rows(1, 2, 3, 4)) == ([], [3])  # appended
    assert run(rows(1, 2, 3, 4), rows(1, 2, 3)) == ([3], [])  # last one removed
    assert run(rows(1, 2, 3, 4), rows(2, 3, 4, 1)) == ([0], [3])  # moved to the end
    assert run(rows(1, 2, 3, 4), rows(2, 1, 3, 4)) in (([0], [1]), ([1], [0]))  # a swap: either row counts as moved
    assert run(rows(), rows(1, 2)) == ([], [0, 1])
    assert run(rows(1, 2), rows()) == ([0, 1], [])
    assert run(rows(), rows()) == ([], [])
    assert run(rows(5, 5, 5), rows(5, 5)) == ([2], [])  # duplicated rows
    assert function(rows(1, 2, 3, 4), rows(5, 6, 7, 8), 3)[2] == -1  # more than ``limit`` unpaired rows
    assert function(rows(1, 2, 3, 4), rows(1), 2)[2] == -1
    assert function(rows(1), rows(5, 6, 7), 2)[2] == -1  # ... or on the child side


def test_unmatched_rows_is_always_a_valid_pairing():
    """Whatever the changes, the rows left out are the only ones that can differ: the paired ones are equal."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        parent = rng.integers(0, 6, size=(rng.integers(0, 9), 2)).astype(np.float32)
        child = rng.integers(0, 6, size=(rng.integers(0, 9), 2)).astype(np.float32)
        old, new, n_old, n_new = kernels.unmatched_rows(parent, child, 64)
        keep_old = [parent[i] for i in range(len(parent)) if i not in set(old[:n_old])]
        keep_new = [child[j] for j in range(len(child)) if j not in set(new[:n_new])]
        assert len(keep_old) == len(keep_new)
        assert all((a == b).all() for a, b in zip(keep_old, keep_new, strict=True))


@pytest.mark.parametrize("function", [kernels.merge_boxes, kernels.merge_boxes.py_func])
def test_merge_boxes_joins_overlapping_boxes_and_drops_empty_ones(function):
    def run(*boxes):
        merged = function(np.array(boxes, dtype=np.int64).reshape(-1, 4))
        return sorted(map(tuple, merged.tolist()))

    assert run((0, 4, 0, 4), (2, 6, 2, 6)) == [(0, 6, 0, 6)]
    assert run((0, 2, 0, 2), (5, 7, 5, 7)) == [(0, 2, 0, 2), (5, 7, 5, 7)]
    assert run((0, 2, 0, 2), (2, 4, 0, 2)) == [(0, 2, 0, 2), (2, 4, 0, 2)]  # touching is not overlapping
    assert run((0, 4, 0, 4), (6, 9, 6, 9), (3, 7, 3, 7)) == [(0, 9, 0, 9)]  # a chain joins through the middle box
    assert run((3, 3, 0, 5), (0, 5, 4, 4), (1, 2, 1, 2)) == [(1, 2, 1, 2)]
    assert run() == []


def test_the_window_kernel_matches_its_python_source():
    canvas = np.full((4, 5, 3), 0.5, dtype=np.float32)
    target = target_image()
    assert kernels.window_error.py_func(canvas, target, 2, 3) == pytest.approx(kernels.window_error(canvas, target, 2, 3))
    assert kernels.window_error(canvas, target, 2, 3) == pytest.approx(float(((canvas - target[3:7, 2:7]) ** 2).sum()), rel=1e-5)


@pytest.mark.parametrize("name", INCREMENTAL)
def test_a_run_reports_the_error_of_the_genome_it_returns(name):
    problem = image_problem(name)
    result = GeneticAlgorithm(problem, population_size=12, max_generations=25, seed=3, mutation_rate=0.3,
                              crossover_rate=0.3, diversity=None).run()
    assert result.best.fitness == pytest.approx(problem.fitness(result.best.genome), rel=1e-6)
    assert all(i.fitness == pytest.approx(problem.fitness(i.genome), rel=1e-6) for i in result.population)


@pytest.mark.parametrize("function", [kernels.covered_pixels, kernels.covered_pixels.py_func])
def test_covered_pixels_adds_up_the_windows(function):
    assert function(np.array([[0, 4, 0, 3], [10, 12, 5, 6]], dtype=np.int64)) == 4 * 3 + 2 * 1
    assert function(np.empty((0, 4), dtype=np.int64)) == 0
