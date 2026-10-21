# ![cmenpy](assets/cmenpy.png){ .cmenpy-hero } { #cmenpy }

**C**ompute **M**etaheuristic **En**gine for **P**ython (_cmenpy_, pronounced
/siːmɛnpaɪ/) is an extensible, vectorized Python framework for designing and
prototyping metaheuristic and population-based optimization algorithms.

```bash
pip install cmenpy          # runtime: numpy only (pure-Python kernels)
pip install "cmenpy[numba]" # recommended: compiles kernels with Numba
```

An optimizer is a function — or a class — that steps a population forward one
epoch at a time. This is a complete one:

```python
import numpy as np
import cmenpy as cm


@cm.declare(x=cm.Argument[int, (0, 3), 2])
def my_algorithm(args, epoch, ctx):
    pop = cm.PopulationSwarm(ctx.buff)
    pop.compute << ctx.rng.uniform(
        ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
    )
    for _ in epoch:
        pop.compute << np.clip(
            ~pop.extract + ctx.rng.uniform(-1, 1, (pop.size, ctx.bounds.ndim)),
            ctx.bounds.low,
            ctx.bounds.up,
        )


best_fitness, best_x = my_algorithm(x=1).solve(
    lambda x: np.sum(x**2), cm.Bounds([(-1, 1)] * 3), epochs=150, pop_size=15
)
```

`solve` runs the model against any objective and returns the best solution
found and its fitness.

## What you get

- **Flat populations.** Solutions live in one NumPy array — the fitness column
  and the vectors share a single buffer. No per-individual objects.
- **Two declaration styles.** `@cm.declare` wraps a function that drives the run;
  `@cm.template` wraps a class with `initialize`/`evolve`. Both produce the same
  `solve` contract.
- **Reusable operator modules.** `@cm.module` bundles kernels, helpers and
  `cm.Constant` fields that any number of models inherit.
- **Compiled kernels.** `@cm.precompile`, `@cm.vectorize` and `@cm.reduce` defer
  Numba compilation to first use and cast by annotation.
- **Benchmark ready.** Reference test functions plug in directly from the
  LTSIM companions [`opfunu-core`](https://github.com/ltsim/opfunu-core) and
  [`enoppy-core`](https://github.com/ltsim/enoppy-core) — see
  [Test functions](benchmarks.md).

Start with the [Quickstart](quickstart.md), benchmark with
[Test functions](benchmarks.md), build your own optimizer with
[Build a custom model](tutorial-custom-model.md), or look things up in the
[API reference](api/index.md).

*Licensed under [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0). Developed by
[LTSIM](mailto:tsim@cucei.udg.mx).*
