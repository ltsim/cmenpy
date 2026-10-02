import numpy as np
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
            w_pop = pop.F[pop.worst]

            pop.remove(pop.worst)
        else:
            if pop.free_space:
                n_pop = pop.insert(rng.uniform(-1, 1, bounds.ndim))

    @cm.vectorize()
    def test_vectorize(x, y):
        return x * 2.0 + y ** 2.0

    @cm.jit()
    def test_jit(A):
        return A ** 2


if __name__ == "__main__":
    model = MyAlgorithm(a=3, b=2, seed=1)

    def sphere(x):
        return np.sum(x**2)

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def ackley(x):
        n = len(x)
        s1 = np.sum(x**2)
        s2 = np.sum(np.cos(2 * np.pi * x))

        return -20 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e

    def peaks(i):
        x = i[0]
        y = i[1]

        a = 3 * (1 - x) ** 2 * np.exp(-(x**2) - (y + 1) ** 2)
        b = -10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2) - y**2)
        c = -1 / 3 * np.exp(-((x + 1) ** 2) - y**2)

        return a + b + c

    best_pop = model.solve(
        sphere, cm.Bounds[(-1, 1), (-1, 1)], 1500, 15, (1, 20), debug=True
    )
    print("Best population:", best_pop)
