"""Particle Swarm Optimization (PSO) built with the ``cm.template`` class API.

Swarm state (velocity, personal bests and global best) is kept as public class
attributes declared with ``cm.Variable``. The update kernels are compiled with
numba and use the ``cm.Array``/``cm.Scalar`` casting markers:

* ``@cm.jit``       -> matrix algebra for the velocity update.
* ``@cm.vectorize`` -> elementwise position update (clip).
* ``@cm.parallel``  -> ``nb.prange`` velocity clamp.
"""

import numba as nb
import numpy as np
from opfunu.name_based import Ackley01

import cmenpy as cm


@cm.template
class ParticleSwarmOptimization:
    w: cm.Argument[float, (0.0, 1.0), 0.72]
    c1: cm.Argument[float, (0.0, 4.0), 1.49]
    c2: cm.Argument[float, (0.0, 4.0), 1.49]

    v: cm.Variable[np.ndarray]
    pbest: cm.Variable[np.ndarray]
    pbest_f: cm.Variable[np.ndarray]
    gbest: cm.Variable[np.ndarray]

    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        span = bounds.up - bounds.low
        X = rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

        self.v = 0.1 * rng.uniform(-span, span, (pop.size, bounds.ndim))
        self.pbest = X.copy()

        pop.compute << X
        self.pbest_f = np.array(pop.F)[: pop.size].copy()
        self.gbest = self.pbest[cm.best_index(self.pbest_f, ctx.sense)].copy()

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        X = pop.X[: pop.size]
        r1 = rng.random((pop.size, bounds.ndim))
        r2 = rng.random((pop.size, bounds.ndim))

        # V = w*V + c1*r1*(P - X) + c2*r2*(G - X), then clamped to vmax.
        self.v = self.limit(
            self.velocity(
                X, self.v, self.pbest, self.gbest, r1, r2,
                self.w, self.c1, self.c2,
            ),
            0.5 * (bounds.up - bounds.low),
        )
        X = self.position(X, self.v, bounds.low, bounds.up)

        pop.compute << X
        F = np.array(pop.F)[: pop.size]

        improved = cm.is_best((F, self.pbest_f), ctx.sense)

        if len(improved):
            self.pbest[improved] = X[improved]
            self.pbest_f[improved] = F[improved]

        self.gbest = self.pbest[cm.best_index(self.pbest_f, ctx.sense)].copy()

    @cm.jit(
        cm.Array[float, 2],
        cm.Array[float, 2],
        cm.Array[float, 2],
        cm.Array[float, 1],
        cm.Array[float, 2],
        cm.Array[float, 2],
        cm.Scalar[float],
        cm.Scalar[float],
        cm.Scalar[float],
    )
    def velocity(x, v, pbest, gbest, r1, r2, w, c1, c2):
        return w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)

    @cm.vectorize(
        cm.Scalar[
            float,
            [
                cm.Scalar[float],
                cm.Scalar[float],
                cm.Scalar[float],
                cm.Scalar[float],
            ],
        ]
    )
    def position(x, v, low, up):
        return min(max(x + v, low), up)

    @cm.parallel(cm.Array[float, 2], cm.Scalar[float])
    def limit(v, vmax):
        out = np.empty_like(v)

        for i in nb.prange(v.shape[0]):
            for j in range(v.shape[1]):
                out[i, j] = min(max(v[i, j], -vmax), vmax)

        return out


if __name__ == "__main__":
    model = ParticleSwarmOptimization(seed=1)
    f = Ackley01(ndim=30)

    best_pop = model.solve(f.evaluate, f.bounds, 300, 40)
    print("Best population:", best_pop)
