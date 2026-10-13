import numpy as np
import pytest

import cmenpy as cm
from cmenpy.target import MaxNumberOfCalls, TargetFunction


def sphere(x):
    return float(np.sum(x**2))


def sphere_matrix(x):
    return np.sum(x**2, axis=1)


BOUNDS = cm.Bounds[(-1, 1), (-1, 1)]


def test_plain_target_is_not_vectorized() -> None:
    target = TargetFunction(sphere, BOUNDS)

    assert target.vectorized is False


def test_wrapped_target_is_vectorized() -> None:
    target = TargetFunction(cm.vectorized_target(sphere_matrix), BOUNDS)

    assert target.vectorized is True


def test_evaluate_matrix_rejects_plain_target() -> None:
    target = TargetFunction(sphere, BOUNDS)

    with pytest.raises(TypeError):
        target.evaluate_matrix(np.zeros((3, 2)))


def test_evaluate_matrix_matches_row_by_row() -> None:
    x = np.array([[1.0, 2.0], [-3.0, 0.5], [0.0, 0.0]])

    vectorized = TargetFunction(cm.vectorized_target(sphere_matrix), BOUNDS)
    plain = TargetFunction(sphere, BOUNDS)

    np.testing.assert_allclose(
        vectorized.evaluate_matrix(x), [plain.evaluate(row) for row in x]
    )


def test_vectorized_call_counts_every_individual() -> None:
    target = TargetFunction(cm.vectorized_target(sphere_matrix), BOUNDS)

    target.evaluate_matrix(np.zeros((4, 2)))

    assert target.nfe == 4


def test_max_nfe_is_enforced_per_individual() -> None:
    target = TargetFunction(cm.vectorized_target(sphere_matrix), BOUNDS, 4)

    target.evaluate_matrix(np.zeros((3, 2)))
    assert target.nfe == 3

    with pytest.raises(MaxNumberOfCalls):
        target.evaluate_matrix(np.zeros((3, 2)))


def test_max_nfe_is_enforced_row_by_row() -> None:
    target = TargetFunction(sphere, BOUNDS, 1)

    target.evaluate(np.zeros(2))
    assert target.nfe == 1

    with pytest.raises(MaxNumberOfCalls):
        target.evaluate(np.zeros(2))


def test_class_with_marker_attribute_is_vectorized() -> None:
    class Sphere:
        vectorized = True

        def __call__(self, x):
            return np.sum(x**2, axis=1)

    target = TargetFunction(Sphere(), BOUNDS)

    assert target.vectorized is True


def test_truthy_marker_is_not_enough() -> None:
    # The opt-in has to be explicit: an accidental truthy attribute must not
    # silently switch the buffer to a single-call evaluation.
    class Sphere:
        vectorized = "yes"

        def __call__(self, x):
            return np.sum(x**2, axis=1)

    target = TargetFunction(Sphere(), BOUNDS)

    assert target.vectorized is False


@cm.declare(...)
def minimize(args, epoch, ctx):
    pop = cm.PopulationSwarm(ctx.buff)
    n_pop = cm.PopulationSwarm(ctx.buff.snapshot)
    rng = ctx.rng

    pop.compute << rng.uniform(-1.0, 1.0, (pop.size, ctx.bounds.ndim))

    for e in epoch:
        n_pop.compute << (pop.X[: pop.size] - 0.1 * e)

        mask = cm.is_best((n_pop.F, pop.F), ctx.sense)
        pop.assign[mask] << n_pop.access[mask]


def solve_with(target) -> tuple[float, np.ndarray]:
    return minimize(seed=7).solve(target, BOUNDS, 5, 6)


def test_vectorized_target_reaches_the_same_optimum() -> None:
    scalar_fitness, scalar_x = solve_with(sphere)
    vector_fitness, vector_x = solve_with(cm.vectorized_target(sphere_matrix))

    assert scalar_fitness == pytest.approx(vector_fitness)
    np.testing.assert_allclose(scalar_x, vector_x)


def test_buffer_dispatches_vectorized_targets_once() -> None:
    calls: list[int] = []

    def counting_matrix(x):
        calls.append(x.shape[0])

        return sphere_matrix(x)

    minimize(seed=7).solve(cm.vectorized_target(counting_matrix), BOUNDS, 5, 6)

    # One call per population, not one per individual.
    assert calls == [6] * 6
