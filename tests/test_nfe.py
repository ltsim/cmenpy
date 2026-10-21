"""NFE accounting for every selection form ``compute`` scores through.

The counter must spend exactly one evaluation per *individual scored*,
whether the block is written in its entirety, through a mask, or one row at a
time (including NumPy row indices such as ``pop.best``).
"""

import typing

import numpy as np
import pytest

import cmenpy as cm
from cmenpy.population.agent import MutableAgent
from cmenpy.target import MaxNumberOfCalls, TargetFunction

BOUNDS = cm.Bounds[(-1, 1), (-1, 1)]


class CountingTarget:
    """Objective that records how many individuals each call actually scored.

    Args:
        block: Score whole ``(k, ndim)`` blocks (the vectorized contract)
            instead of single solution vectors.
    """

    def __init__(self, block: bool) -> None:
        """Store the scoring mode and reset the counter."""
        self.block = block
        self.scored = 0

    def __call__(self, x: typing.Any) -> typing.Any:
        """Score ``x`` and add its individuals to the observed counter."""
        array = np.asarray(x)
        self.scored += len(array) if array.ndim > 1 else 1

        if self.block:
            return np.sum(array**2, axis=1)

        return float(np.sum(array**2))


def _swarm(
    block: bool,
    size: int = 4,
    pop_range: tuple[int, int] | None = None,
) -> tuple[cm.PopulationSwarm, cm.KernelBuffer, TargetFunction, CountingTarget]:
    """Return a swarm and its buffer over a counting target and its counter."""
    counter = CountingTarget(block)
    target = cm.vectorized_target(counter) if block else counter
    tf = TargetFunction(target, BOUNDS)
    min_size, max_size = pop_range or (size, size)
    buff = cm.KernelBuffer(cm.KernelSize(size, min_size, max_size), BOUNDS, tf, "min")

    return cm.PopulationSwarm(buff), buff, tf, counter


@pytest.mark.parametrize("block", [False, True])
def test_compute_entirety_scores_active_rows_only(block: bool) -> None:
    """Writing without an index spends one evaluation per active row."""
    pop, _, tf, counter = _swarm(block, pop_range=(2, 4))
    pop.remove(1)
    pop.remove(3)

    pop.compute << np.full((2, 2), 0.5)

    assert tf.nfe == 2 == counter.scored
    assert np.isnan(pop.F[1]) and np.isnan(pop.F[3])


@pytest.mark.parametrize("block", [False, True])
def test_compute_with_boolean_mask_scores_selected_rows(block: bool) -> None:
    """A boolean mask spends one evaluation per selected row, no more."""
    pop, _, tf, counter = _swarm(block)

    pop.compute[np.array([True, False, True, False])] << np.full((2, 2), 0.5)

    assert tf.nfe == 2 == counter.scored
    assert np.isnan(pop.F[1]) and np.isnan(pop.F[3])


@pytest.mark.parametrize("block", [False, True])
def test_compute_with_mask_object_scores_active_rows(block: bool) -> None:
    """The buffer's own ``KernelMask`` selects exactly the active rows."""
    pop, buff, tf, counter = _swarm(block, pop_range=(2, 4))
    pop.remove(2)

    pop.compute[buff.mask] << np.full((3, 2), 0.5)

    assert tf.nfe == 3 == counter.scored


@pytest.mark.parametrize("block", [False, True])
def test_compute_with_index_array_scores_selected_rows(block: bool) -> None:
    """Integer index arrays spend one evaluation per listed row."""
    pop, _, tf, counter = _swarm(block)

    pop.compute[np.array([3, 0])] << np.full((2, 2), 0.5)

    assert tf.nfe == 2 == counter.scored


@pytest.mark.parametrize("block", [False, True])
def test_compute_with_slice_scores_selected_rows(block: bool) -> None:
    """A slice spends one evaluation per row it selects."""
    pop, _, tf, counter = _swarm(block)

    pop.compute[1:] << np.full((3, 2), 0.5)

    assert tf.nfe == 3 == counter.scored


@pytest.mark.parametrize("block", [False, True])
def test_compute_one_row_scores_one_evaluation(block: bool) -> None:
    """A Python row index writes and scores a single individual."""
    pop, _, tf, counter = _swarm(block)

    pop.compute[2] << np.full(2, 0.5)

    assert tf.nfe == 1 == counter.scored
    assert pop.F[2] == pytest.approx(0.5)


@pytest.mark.parametrize("block", [False, True])
def test_compute_one_numpy_row_scores_one_evaluation(block: bool) -> None:
    """A NumPy row index (``pop.best``) costs one evaluation, not ``ndim``."""
    pop, _, tf, counter = _swarm(block)
    pop.compute << np.arange(8, dtype=float).reshape(4, 2)

    pop.compute[pop.best] << np.zeros(2)

    assert tf.nfe == 5 == counter.scored
    assert pop.F[pop.best] == pytest.approx(0.0)


@pytest.mark.parametrize("block", [False, True])
def test_compute_numpy_row_matches_python_row(block: bool) -> None:
    """The same row costs one evaluation whichever integer flavour is used."""
    numpy_pop, _, numpy_tf, _ = _swarm(block)
    python_pop, _, python_tf, _ = _swarm(block)
    numpy_pop.compute << np.zeros((4, 2))
    python_pop.compute << np.zeros((4, 2))

    numpy_pop.compute[numpy_pop.best] << np.full(2, 0.5)
    python_pop.compute[int(python_pop.best)] << np.full(2, 0.5)

    assert numpy_tf.nfe == python_tf.nfe == 5


@pytest.mark.parametrize("block", [False, True])
def test_compute_with_active_index_iterator_scores_selected_rows(
    block: bool,
) -> None:
    """``~pop`` (the scored-row index iterator) is a valid selection."""
    pop, _, tf, counter = _swarm(block)
    pop.compute << np.zeros((4, 2))

    pop.compute[~pop] << np.full((4, 2), 0.5)

    assert tf.nfe == 8 == counter.scored


@pytest.mark.parametrize("block", [False, True])
def test_mismatched_block_spends_nothing(block: bool) -> None:
    """A block that cannot fit the selection scores nothing."""
    pop, _, tf, counter = _swarm(block)

    with pytest.raises(ValueError):
        pop.compute[[0, 1]] << np.full((3, 2), 0.5)

    assert tf.nfe == 0 == counter.scored


@pytest.mark.parametrize("block", [False, True])
def test_insert_scores_one_evaluation(block: bool) -> None:
    """``insert`` costs a single evaluation."""
    pop, _, tf, counter = _swarm(block, size=2, pop_range=(2, 4))
    pop.compute << np.zeros((2, 2))

    idx = pop.insert(np.full(2, 0.5))

    assert idx == 2
    assert tf.nfe == 3 == counter.scored
    assert pop.F[idx] == pytest.approx(0.5)


@pytest.mark.parametrize("block", [False, True])
def test_agent_solution_write_scores_one_evaluation(block: bool) -> None:
    """Re-writing an agent's solution costs a single evaluation."""
    pop, buff, tf, counter = _swarm(block)
    pop.compute << np.arange(8, dtype=float).reshape(4, 2)
    agent = MutableAgent(pop.best, buff)

    agent.solution = np.full(2, 0.5)

    assert tf.nfe == 5 == counter.scored
    assert pop.F[agent.i] == pytest.approx(0.5)


@pytest.mark.parametrize("block", [False, True])
def test_apply_axis_scores_one_evaluation_per_selected_row(block: bool) -> None:
    """The masked row-wise helper spends one evaluation per selected row."""
    counter = CountingTarget(block)
    target = cm.vectorized_target(counter) if block else counter
    tf = TargetFunction(target, BOUNDS)
    raw = np.zeros((4, 3))

    scores = tf.apply_axis(raw, np.array([0, 2]))

    assert scores.shape == (2,)
    assert tf.nfe == 2 == counter.scored


def test_vectorized_row_and_block_agree_on_the_counter() -> None:
    """A single row and its one-row block both cost exactly one evaluation."""
    counter = CountingTarget(True)
    tf = TargetFunction(cm.vectorized_target(counter), BOUNDS)

    row = tf.evaluate(np.full(2, 0.5))
    block_score = tf.evaluate_matrix(np.full((1, 2), 0.5))

    assert row == pytest.approx(0.5)
    assert block_score.tolist() == [0.5]
    assert tf.nfe == 2 == counter.scored


def test_evaluate_matrix_rejects_a_single_row() -> None:
    """A bare solution vector is not a block; it must not spend ``ndim``."""
    tf = TargetFunction(cm.vectorized_target(CountingTarget(True)), BOUNDS)

    with pytest.raises(TypeError):
        tf.evaluate_matrix(np.zeros(2))

    assert tf.nfe == 0


def test_evaluate_never_under_counts_a_block() -> None:
    """Routing a multi-row block through ``evaluate`` still charges every row."""
    counter = CountingTarget(True)
    tf = TargetFunction(cm.vectorized_target(counter), BOUNDS)

    scores = tf.evaluate(np.full((3, 2), 0.5))

    assert scores.tolist() == [0.5, 0.5, 0.5]
    assert tf.nfe == 3 == counter.scored


def test_budget_charges_one_unit_per_individual_write() -> None:
    """Individual writes stay inside ``max_nfe`` exactly one unit at a time."""
    counter = CountingTarget(True)
    tf = TargetFunction(cm.vectorized_target(counter), BOUNDS, max_nfe=8)
    pop = cm.PopulationSwarm(cm.KernelBuffer(cm.KernelSize(4), BOUNDS, tf, "min"))

    for i in range(8):
        pop.compute[np.int64(i % 4)] << np.full(2, 0.5)

    assert tf.nfe == 8 == counter.scored

    with pytest.raises(MaxNumberOfCalls):
        pop.compute[np.int64(0)] << np.full(2, 0.5)


@cm.declare(...)
def nfe_counter(args, epoch, ctx):
    """Score the whole block per epoch plus one individually written row."""
    pop = cm.PopulationSwarm(ctx.buff)

    pop.compute << np.zeros((pop.size, ctx.bounds.ndim))

    for _ in epoch:
        pop.compute << np.full((pop.size, ctx.bounds.ndim), 0.5)
        pop.compute[pop.best] << np.full(ctx.bounds.ndim, 0.25)


@pytest.mark.parametrize("block", [False, True])
def test_run_history_counts_each_individual_once(block: bool) -> None:
    """The tracked NFE grows by ``pop_size + 1`` per epoch, in both modes."""
    counter = CountingTarget(block)
    target = cm.vectorized_target(counter) if block else counter
    model = nfe_counter(seed=1)

    model.solve(target, BOUNDS, epochs=2, pop_size=3, debug=True)

    assert [history.nfe for history in model.tracker] == [7, 11]
    assert counter.scored == 11
