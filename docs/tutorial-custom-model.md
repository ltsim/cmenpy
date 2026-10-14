# Build a custom model

This tutorial walks through implementing a custom optimizer from scratch with
the `@cm.template` class style. We will build a **(μ + λ) Evolution Strategy**
(Es): keep a parent population of size μ, produce λ offspring by Gaussian
mutation each epoch, and select the best μ from the union of parents and
offspring (elitist selection).

Along the way you will use every core piece of `cmenpy`: fields, the `ctx`
object, the [Population](population.md) DSL, the [comparators](api/utilities.md),
and a compiled [numba kernel](compile.md).

!!! info "Prerequisites"
    Comfortable with NumPy. You should have skimmed the
    [Quickstart](quickstart.md) and [Models](models.md) overview.

## What we are building

```python
import numpy as np
import cmenpy as cm

@cm.template
class EvolutionStrategy:
    sigma: cm.Argument[float, (1e-6, 10.0), 0.5]   # mutation step (initial)

    def initialize(self, ctx) -> None: ...
    def evolve(self, e, ctx) -> None: ...

model = EvolutionStrategy(seed=42)
best_fitness, best_x = model.solve(
    lambda x: np.sum(x**2), cm.Bounds([(-1, 1)] * 5), epochs=100, pop_size=30
)
```

Let's build it step by step.

## Step 1 — Declare the class and fields

Start with the skeleton. Class annotations declare tunable parameters with
`cm.Argument[type, (min, max), default]` and free state with `cm.Variable[type]`:

```python
@cm.template
class EvolutionStrategy:
    sigma: cm.Argument[float, (1e-6, 10.0), 0.5]

    offspring: cm.Variable[np.ndarray]
```

- `sigma` is a **tunable argument**: range-checked, with a default, and settable
  at construction (`EvolutionStrategy(sigma=0.1)`).
- `offspring` is a **variable**: internal state with no range or default. We
  will store the current generation's offspring here so `evolve` can both
  produce and score them.

A `@cm.template` model must implement two methods: `initialize(self, ctx)` (runs
once) and `evolve(self, e, ctx)` (runs once per epoch). Let's add them next.

## Step 2 — Seed the population in `initialize`

`initialize` runs before any epoch. Build a `PopulationSwarm` over the buffer
and write an initial random population with `pop.compute << ...`, which both
inserts **and scores** the solutions:

```python
def initialize(self, ctx) -> None:
    pop = cm.PopulationSwarm(ctx.buff)
    pop.compute << ctx.rng.uniform(
        ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
    )
```

- `ctx.buff` is the flat buffer backing the population.
- `ctx.bounds.low` / `ctx.bounds.up` are the box bounds — convenient for
  vectorized sampling.
- `pop.size` is the number of active individuals.

After this line the whole parent population is randomly placed and evaluated.

## Step 3 — Produce offspring in `evolve`

`evolve(self, e, ctx)` is called once per epoch with the epoch index `e`. We
mutate μ randomly chosen parents λ times. First, grab the active parent
solutions (`~pop.extract`), then sample parent indices and perturb:

```python
def evolve(self, e, ctx) -> None:
    pop = cm.PopulationSwarm(ctx.buff)
    parents = ~pop.extract                       # (μ, ndim) active solutions

    # Anneal the step size linearly over the run.
    self.offspring = self.sigma * (1.0 - e / ctx.epochs)

    idx = ctx.rng.integers(0, pop.size, pop.size)  # λ = μ
    kids = parents[idx]                            # pick parents at random

    noise = ctx.rng.standard_normal(kids.shape)
    mutated = np.clip(
        kids + self.offspring * noise,
        ctx.bounds.low,
        ctx.bounds.up,
    )
```

We reuse the buffer's snapshot to score the offspring in isolation, without
clobbering the parents yet:

```python
    n_pop = cm.PopulationSwarm(ctx.buff.snapshot)
    n_pop.compute << mutated
```

!!! note "`ctx.buff.snapshot`"
    `snapshot` returns a fresh, empty buffer of the same shape wired to the same
    objective and sense. It is the idiomatic place to evaluate a candidate
    generation before merging it back.

## Step 4 — Elitist selection with `cm.is_best`

Now merge: keep, for each row, whichever of {parent, offspring} is better. The
`cm.is_best((new_F, old_F), ctx.sense)` comparator returns the row indices where
`new` beats `old`, respecting the optimization direction:

```python
    mask = cm.is_best((n_pop.F, pop.F), ctx.sense)
    pop.assign[mask] << n_pop.access[mask]
```

- `n_pop.F` / `pop.F` are the fitness columns of the offspring and parents.
- `pop.assign[mask] << n_pop.access[mask]` copies the winning offspring rows
  over their parents (`access` reads whole rows, `assign` writes them).

Because we never introduce a worse individual, the best solution is monotonically
non-worsening — classic elitism. Note we used `ctx.sense` rather than a hard-coded
comparison, so this model works for `"min"` and `"max"` alike.

## Step 5 — Compile the mutation with a numba kernel

The mutation loop above is plain NumPy. To make it run at numba speed, move the
per-gene math into a `@cm.precompile` kernel declared as a method. Annotate the
parameters so `cmenpy` knows how to cast and type them:

```python
@cm.template
class EvolutionStrategy:
    sigma: cm.Argument[float, (1e-6, 10.0), 0.5]
    offspring: cm.Variable[np.ndarray]

    @cm.precompile
    def mutate(
        K: "Array[float, 2]",
        N: "Array[float, 2]",
        s: "Scalar[float]",
        lo: "Scalar[float]",
        hi: "Scalar[float]",
    ):
        out = np.empty_like(K)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                v = K[i, j] + s * N[i, j]
                out[i, j] = min(max(v, lo), hi)   # clip to bounds
        return out
```

Then call it from `evolve` instead of the inline NumPy:

```python
    mutated = self.mutate(kids, noise, self.offspring,
                          ctx.bounds.low, ctx.bounds.up)
```

Because `mutate` is a method on a `@cm.template` class, it participates in the
model's compile options — you can force `parallel`, `fastmath` or `cache` at
construction (see [Numba kernels](compile.md#model-compile-options)). The kernel
compiles lazily on first use; call `model.prepare()` to compile eagerly and
inspect the signatures.

## Step 6 — Run it

Wire it all together and solve:

```python
if __name__ == "__main__":
    model = EvolutionStrategy(seed=42)

    def sphere(x):
        return float(np.sum(x**2))

    best_fitness, best_x = model.solve(
        sphere, cm.Bounds([(-1, 1)] * 5), epochs=100, pop_size=30
    )
    print("best fitness:", best_fitness)
    print("best solution:", best_x)
```

`solve` returns `(best_fitness, best_x)`. Pass `debug=True` to populate
`model.tracker` with per-epoch history (`EpochHistory` records), and
`sense="max"` to switch to maximization.

## Full example

```python
import numpy as np
import cmenpy as cm


@cm.template
class EvolutionStrategy:
    """A (mu + lambda) Evolution Strategy with elitist selection."""

    sigma: cm.Argument[float, (1e-6, 10.0), 0.5]

    offspring: cm.Variable[np.ndarray]

    @cm.precompile
    def mutate(
        K: "Array[float, 2]",
        N: "Array[float, 2]",
        s: "Scalar[float]",
        lo: "Scalar[float]",
        hi: "Scalar[float]",
    ):
        out = np.empty_like(K)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                v = K[i, j] + s * N[i, j]
                out[i, j] = min(max(v, lo), hi)
        return out

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(
            ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
        )

    def evolve(self, e, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        parents = ~pop.extract

        self.offspring = self.sigma * (1.0 - e / ctx.epochs)

        idx = ctx.rng.integers(0, pop.size, pop.size)
        kids = parents[idx]
        noise = ctx.rng.standard_normal(kids.shape)

        mutated = self.mutate(
            kids, noise, self.offspring, ctx.bounds.low, ctx.bounds.up
        )

        n_pop = cm.PopulationSwarm(ctx.buff.snapshot)
        n_pop.compute << mutated

        mask = cm.is_best((n_pop.F, pop.F), ctx.sense)
        pop.assign[mask] << n_pop.access[mask]


if __name__ == "__main__":
    model = EvolutionStrategy(seed=42)

    def sphere(x):
        return float(np.sum(x**2))

    best_fitness, best_x = model.solve(
        sphere, cm.Bounds([(-1, 1)] * 5), epochs=100, pop_size=30
    )
    print("best fitness:", best_fitness)
    print("best solution:", best_x)
```

## Where to go next

- **Function style** — prefer `@cm.declare`? See
  [Models → `@cm.declare`](models.md#cmdeclare-over-a-function) and
  `example/hello_world_v1.py`.
- **More operators** — tournament selection, crossover and archives appear in
  the algorithms under `example/` (see [Examples](examples.md)).
- **Vectorized targets** — score a whole population in one call with
  [Target functions](target.md).
- **Kernel recipes** — `@cm.vectorize` for elementwise ufuncs and `@cm.reduce`
  for generalized ufuncs in [Numba kernels](compile.md).
