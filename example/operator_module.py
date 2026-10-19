"""Reuse operators and constants across models with ``@cm.module``.

An Operator Module is a Ruby-style module: a bundle of kernels, plain helper
methods and fields (``cm.Constant`` / ``cm.Argument``) that any number of
``@cm.template`` models can mix in by inheriting from it.
"""

import numpy as np

import cmenpy as cm


@cm.module
class MutationOps:
    """A reusable Gaussian-mutation operator module."""

    scale: cm.Constant[float, 0.3]

    @cm.vectorize
    def clip(x: cm.Array, lo: cm.Scalar, hi: cm.Scalar):
        return min(max(x, lo), hi)

    def mutate(self, parents, rng):
        noise = rng.standard_normal(parents.shape)

        return self.clip(parents + self.scale * noise, -1.0, 1.0)


@cm.template
class HillClimber(MutationOps):
    """Population-based search that reuses ``MutationOps`` unchanged."""

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(-1, 1, (pop.size, ctx.bounds.ndim))

    def evolve(self, e, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        trial = cm.PopulationSwarm(ctx.buff.snapshot)
        trial.compute << self.mutate(~pop.extract, ctx.rng)

        # Keep a child only if it beats its parent (elitism).
        mask = cm.is_best((trial.F, pop.F), ctx.sense)
        pop.assign[mask] << trial.access[mask]


@cm.template
class Annealer(MutationOps):
    """A different model that reuses the same operator, scaling it over time."""

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(-1, 1, (pop.size, ctx.bounds.ndim))

    def evolve(self, e, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        trial = cm.PopulationSwarm(ctx.buff.snapshot)

        # Reuse the inherited constant, but temper it with the epoch progress.
        step = self.scale * (1.0 - e / ctx.epochs)
        kids = ~pop.extract + step * ctx.rng.standard_normal(
            (pop.size, ctx.bounds.ndim)
        )
        trial.compute << self.clip(kids, -1.0, 1.0)

        # Keep a child only if it beats its parent (elitism).
        mask = cm.is_best((trial.F, pop.F), ctx.sense)
        pop.assign[mask] << trial.access[mask]


if __name__ == "__main__":

    def sphere(x):
        return float(np.sum(x**2))

    bounds = cm.Bounds([(-5.0, 5.0)] * 3)

    for optimizer in (HillClimber, Annealer):
        model = optimizer(seed=1)
        best_fitness, best_x = model.solve(sphere, bounds, epochs=60, pop_size=20)
        print(f"{optimizer.alias}: f={best_fitness:.5f} x={best_x}")
        print(f"  constants={optimizer.model_class.scale}")
