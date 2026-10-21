"""Solve a constrained engineering problem with ``enoppy-core``.

The three-bar truss design problem minimizes the cross-sectional area of a
two-variable truss under displacement constraints. ``enoppy`` folds the
constraints into a penalized ``evaluate`` call that returns a one-element
array — the objective wrapper below coerces it to a plain float, which is
what ``cmenpy`` targets expect.

Run with::

    .venv/bin/python example/engineering_demo.py
"""

import numpy as np
from enoppy.paper_based.pdo_2022 import ThreeBarTrussProblem

import cmenpy as cm


@cm.template
class HillClimber:
    """Population-based random hill climbing over the penalized problem."""

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(
            ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
        )

    def evolve(self, e, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        trial = cm.PopulationSwarm(ctx.buff.snapshot)

        step = 0.1 * (1.0 - e / ctx.epochs)
        trial.compute << np.clip(
            ~pop.extract + step * ctx.rng.standard_normal((pop.size, ctx.bounds.ndim)),
            ctx.bounds.low,
            ctx.bounds.up,
        )

        mask = cm.is_best((trial.F, pop.F), ctx.sense)
        pop.assign[mask] << trial.access[mask]


if __name__ == "__main__":
    prob = ThreeBarTrussProblem()

    def objective(x):
        # enoppy's `evaluate` returns a one-element array.
        return float(np.asarray(prob.evaluate(x)).item())

    bounds = cm.Bounds(list(zip(prob.lb, prob.ub)))

    best_fitness, best_x = HillClimber(seed=1).solve(
        objective, bounds, epochs=400, pop_size=30
    )

    print(f"Best penalized objective: {best_fitness:.4f} at {best_x}")
    print(f"Raw objectives:  {prob.get_objs(best_x)}")
    print(f"Constraints:     {prob.get_cons(best_x)}")
