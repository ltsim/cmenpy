import numpy as np
import cmenpy as cm


@cm.define("Random search")
def random_search(pop, bounds, epoch):
    pop @= np.random.uniform(-1, 1, (len(pop), bounds.ndim))

    best_pop = pop.best

    for _ in epoch:
        pop @= np.clip(
            ~pop + np.random.uniform(-1, 1, (len(pop), bounds.ndim)), -1, 1
        )

        if pop.best < best_pop:
            best_pop = pop.best
            worst_pop = pop.worst

            pop.remove(worst_pop.id)

            print("Current population:", len(pop))

    return best_pop


def sphere(x):
    return np.sum(np.square(x))


if __name__ == "__main__":
    random_search(sphere, [(-1, 1), (-1, 1)], 5, 5)
