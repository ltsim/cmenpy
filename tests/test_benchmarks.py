"""Integration tests for the LTSIM benchmark companions.

These keep the patterns documented in `docs/benchmarks.md` honest: an
``opfunu-core`` numerical function and an ``enoppy-core`` engineering
problem, each solved by a tiny `@cm.declare` model.
"""

import numpy as np
import pytest

import cmenpy as cm


@cm.declare(x=cm.Argument[int, (0, 3), 1])
def random_search(args, epoch, ctx):
    pop = cm.PopulationSwarm(ctx.buff)
    pop.compute << ctx.rng.uniform(
        ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
    )
    for _ in epoch:
        pop.compute << np.clip(
            ~pop.extract + ctx.rng.uniform(-1.0, 1.0, (pop.size, ctx.bounds.ndim)),
            ctx.bounds.low,
            ctx.bounds.up,
        )


def test_opfunu_benchmark_runs() -> None:
    pytest.importorskip("opfunu")

    from opfunu.name_based import Ackley01

    f = Ackley01(ndim=4)

    assert float(f.evaluate(np.zeros(4))) == pytest.approx(0.0, abs=1e-9)

    best_f, best_x = random_search(x=1, seed=1).solve(
        f.evaluate, f.bounds, epochs=40, pop_size=12
    )

    assert np.isfinite(best_f)
    assert best_f >= 0.0
    assert best_x.shape == (4,)
    assert np.all((best_x >= np.asarray(f.lb)) & (best_x <= np.asarray(f.ub)))


def test_enoppy_engineering_problem_runs() -> None:
    pytest.importorskip("enoppy")

    from enoppy.paper_based.pdo_2022 import ThreeBarTrussProblem

    prob = ThreeBarTrussProblem()

    def objective(x):
        # enoppy's `evaluate` returns a one-element array.
        return float(np.asarray(prob.evaluate(x)).item())

    bounds = cm.Bounds(list(zip(prob.lb, prob.ub)))

    best_f, best_x = random_search(x=1, seed=3).solve(
        objective, bounds, epochs=40, pop_size=12
    )

    assert np.isfinite(best_f)
    assert best_x.shape == (2,)
    assert np.all((best_x >= np.asarray(prob.lb)) & (best_x <= np.asarray(prob.ub)))
