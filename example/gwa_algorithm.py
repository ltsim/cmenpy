import numpy as np
import cmenpy as cm

from opfunu.name_based import Ackley01


@cm.declare()
def gwo(args, epoch, ctx):
    pop, bounds, rng = ctx.population, ctx.bounds, ctx.rng
    f_target = ctx.target.f

    pop @= rng.uniform(bounds.low, bounds.up, (len(pop), bounds.ndim))

    for e in epoch:
        all_pop = pop.sorted
        best_pop = all_pop[:3]
        new_pop = []

        a = 2 - 2 * int(e) / epoch.max

        A = a * (2 * rng.uniform(size=(pop.size, bounds.ndim)) - 1)
        C = 2 * rng.uniform(size=(pop.size, bounds.ndim))

        for i, p in enumerate(all_pop):
            X = [
                b.solution - A[i] * np.abs(C[i] * b.solution - p.solution)
                for b in best_pop
            ]

            solution = np.clip(np.sum(X, axis=0) / 3, bounds.low, bounds.up)

            new_pop.append(solution)

        for i, p in enumerate(new_pop):
            n_p = f_target(p)

            if n_p < pop[i].fitness:
                pop[i] = [n_p, *p]


if __name__ == "__main__":
    model = gwo(seed=None)
    f = Ackley01(ndim=30)

    b_pop = model.solve(
        f.evaluate, [[-5.12, 5.12] for _ in range(f.ndim)], 130, 75, debug=True
    )
    print(b_pop)
