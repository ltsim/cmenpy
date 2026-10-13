# Quickstart

## Function style: `@cm.declare`

```python
import numpy as np
import cmenpy as cm


@cm.declare(x=cm.Argument[int, (0, 3), 2])
def my_algorithm(args, epoch, ctx):
    pop = cm.PopulationSwarm(ctx.buff)
    pop.compute << ctx.rng.uniform(
        ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
    )
    # ... one epoch of your operator ...


best = my_algorithm(x=1).solve(
    lambda x: np.sum(x**2), cm.Bounds([(-1, 1)] * 3), epochs=150, pop_size=15
)
```

Full version: `example/hello_world_v1.py`.

## Class style: `@cm.template`

```python
@cm.template
class MyAlgorithm:
    a: cm.Argument[int, (1, 100)]

    w: cm.Variable[np.array]

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(-1, 1, (pop.size, ctx.bounds.ndim))

    def evolve(self, e, ctx) -> None:
        # ... generation `e` ...
        ...


model = MyAlgorithm(a=3, seed=1)
best = model.solve(lambda x: np.sum(x**2), cm.Bounds([(-1, 1)] * 3), 150, 15)
```

Full version: `example/hello_world_v2.py`.

## The `solve` contract

```python
(best_fitness, best_x) = model.solve(
    f, bounds, epochs, pop_size, pop_range=None, debug=False, sense="min"
)
```

- `f` is a callable `x -> float` over one solution vector (or a
  [vectorized target](target.md)).
- `bounds` is a `cm.Bounds` (e.g. `cm.Bounds([(-1, 1)] * ndim)`).
- `sense` is `"min"` (default) or `"max"`.

## Run the examples

`example/` is not a declared dependency target and is excluded from the wheel.
Some examples import `opfunu`, which must be installed separately:

```bash
uv pip install opfunu
.venv/bin/python -m example.benchmark_gwo   # (if present in your checkout)
.venv/bin/python example/reduce_demo.py
```
