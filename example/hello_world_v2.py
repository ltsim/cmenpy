import numpy as np
from opfunu.name_based import Ackley01

import cmenpy as cm


@cm.template
class MyAlgorithm:
    a: cm.Argument[int, (1, 100)]
    b: cm.Argument[int, (1, 100)]

    w: cm.Variable[np.array]

    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        pop.compute << rng.uniform(-1, 1, (pop.size, bounds.ndim))

        M = np.array([1, 2, 3])
        r = self.test_vectorize(M, M)
        print(r)

        r = self.test_jit(M)
        print(r)

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng

        pop = cm.PopulationSwarm(buff)
        b_pop = pop.F[pop.best]

        pop.compute << (
            np.clip(
                ~pop.extract + rng.uniform(-1, 1, (pop.size, bounds.ndim)),
                -1,
                1,
            )
        )

        if pop.F[pop.best] < b_pop and pop.size > 1:
            pop.remove(pop.worst)
        else:
            if pop.free_space:
                pop.insert(rng.uniform(-1, 1, bounds.ndim))

    @cm.vectorize
    def test_vectorize(x: "Scalar[float]", y: "Scalar[float]") -> "Scalar[float]":
        return x * 2.0 + y**2.0

    @cm.precompile
    def test_jit(A: "Array[float, 1]"):
        return A**2


if __name__ == "__main__":
    model = MyAlgorithm(a=3, b=2, seed=1)

    f = Ackley01(ndim=2)

    best_pop = model.solve(f.evaluate, f.bounds, 1500, 15, (1, 20), debug=True)
    print("Best population:", best_pop)
