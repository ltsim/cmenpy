import numpy as np
import cmenpy as cm

from opfunu.name_based import Ackley01


@cm.declare(...)
def gwo(args, epoch, ctx):
    buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng

    pop = cm.PopulationSwarm(buff)
    pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

    X = np.full((pop.size, bounds.ndim), np.nan)

    for e in epoch:
        n_pop = cm.PopulationSwarm(buff.snapshot)
        b_pop = pop.X[pop.sort][:3]

        a = 2 - 2 * e / epoch.max

        A = a * (2 * rng.random(size=(pop.size, len(b_pop), bounds.ndim)) - 1)
        C = 2 * rng.random(size=(pop.size, len(b_pop), bounds.ndim))

        for i, p in zip(pop.sort, pop.X[pop.sort]):
            D = [b - A[i][j] * np.abs(C[i][j] * b - p) for j, b in enumerate(b_pop)]

            X[i] = np.sum(D, axis=0) / len(b_pop)

        n_pop.compute << np.clip(
            X,
            bounds.low,
            bounds.up,
        )

        mask = np.array(pop.F) > np.array(n_pop.F)
        mask = np.flatnonzero(mask)
        pop.assign[mask] << n_pop.access[mask]


if __name__ == "__main__":
    model = gwo()
    f = Ackley01(ndim=30)

    b_pop = model.solve(f.evaluate, f.bounds, 150, 75)
    print(b_pop)
