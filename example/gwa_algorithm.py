import numpy as np
import cmenpy as cm

from opfunu.name_based import Ackley01


@cm.declare(...)
def gwo(args, epoch, ctx):
    pop, bounds, rng = ctx.population, ctx.bounds, ctx.rng

    pop.compute << rng.uniform(bounds.low, bounds.up, (len(pop), bounds.ndim))

    for e in epoch:
        all_pop = cm.sort_agents(pop)
        new_pop = pop.view

        best_pop = all_pop[:3]

        a = 2 - 2 * e / epoch.max

        A = a * (2 * rng.uniform(size=(pop.size, bounds.ndim)) - 1)
        C = 2 * rng.uniform(size=(pop.size, bounds.ndim))

        X_n = np.zeros((pop.size, bounds.ndim))

        for i, p in enumerate(all_pop):
            xn = [
                b.solution - A[i] * np.abs(C[i] * b.solution - p.solution)
                for b in best_pop
            ]

            X_n[i] = np.sum(xn, axis=0) / 3

        new_pop.compute << np.clip(
            X_n,
            bounds.low,
            bounds.up,
        )

        for i, (a, b) in enumerate(zip(new_pop, all_pop)):
            if b := cm.best_of((a, b)):
                pop.assign[i] << b.buff


if __name__ == "__main__":
    model = gwo()
    f = Ackley01(ndim=30)

    b_pop = model.solve(
        f.evaluate, [[-5.12, 5.12] for _ in range(f.ndim)], 130, 75, debug=True
    )
    print(b_pop)
