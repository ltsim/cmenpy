# Function style (`@cm.declare`)

`@cm.declare` turns a function into a model. The function receives
`(args, epoch, ctx)` and drives the **whole run** inside its body — you
iterate the epochs yourself.

```python
import numpy as np

import cmenpy as cm


@cm.declare(x=cm.Argument[int, (0, 3), 2])
def my_algorithm(args, epoch, ctx):
    pop = cm.PopulationSwarm(ctx.buff)
    pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                                   (pop.size, ctx.bounds.ndim))

    for _ in epoch:                       # one epoch per iteration
        pop.compute << np.clip(
            ~pop.extract
            + ctx.rng.uniform(-1, 1, (pop.size, ctx.bounds.ndim)) * args.x,
            ctx.bounds.low, ctx.bounds.up,
        )
```

The decorated name **is** the model (`FunctionOptimizerModel`).

## The function contract

The function is called **by keyword** with exactly three parameters:

| Parameter | What it is |
| --------- | ---------- |
| `args` | the bound field values (`args.x`, `args.lr`, …) |
| `epoch` | an `EpochIteration` — iterate it to run epochs |
| `ctx` | the per-run [`Context`](index.md#the-ctx-object) |

Name the parameters exactly `args`, `epoch`, `ctx` — the runner passes them
by keyword.

### `epoch`

```python
for _ in epoch:            # yields 0, 1, ..., epochs-1
    ...

epoch.max                  # total epochs
epoch.current              # index of the epoch most recently yielded
epoch.cpu_time             # wall CPU time spent so far
```

The iterator tracks per-epoch CPU time, and with `debug=True` also records a
history snapshot per epoch.

### `ctx`

One `Context` is built **before** your function runs and lives for the whole
run. `ctx.epoch` is therefore fixed at `0` in this style — read progress from
the `epoch` iterator instead.

## Declaring and binding parameters

```python
@cm.declare(
    x=cm.Argument[int, (0, 3), 2],        # range-checked with a default
    lr=cm.Argument[float, (0.0, 1.0)],    # required at binding time
)
def my_algorithm(args, epoch, ctx): ...
```

Declare fields **by keyword** — positional declarations passed to the
decorator (or values passed positionally at binding) are ignored by the
current implementation.

```python
model = my_algorithm(seed=7, x=1, lr=0.1)   # seed first, then fields
model.solve(f, bounds, epochs=100, pop_size=20)
```

- `seed` is the first parameter of the binding call; it seeds the run's
  `ctx.rng`.
- Every field **with a range and no default** must be bound; a field that was
  declared but never bound raises `AttributeError` when read through `args`.
- Only `cm.Argument` is supported — see [Fields](fields.md). `cm.Constant`
  raises at declaration time (it is a class-style feature).

### Errors at binding

| Error | When |
| ----- | ---- |
| `AttributeError: Cannot initialize 'y': It was not declared in MyArguments.` | binding an undeclared name |
| `TypeError: Invalid type in arguments: Invalid type for 'x'. Expected int, got str` | type mismatch |
| `ValueError: Invalid value in arguments: Value 5 for 'x' out of range [0, 3].` | outside the range |
| `AttributeError: Attribute 'lr' has been declared but not initialized yet.` | reading an unbound field through `args` |

Range checks also cover arrays and collections (every element must be in
range).

## Running

```python
best_fitness, best_x = model.solve(
    f, bounds, epochs=100, pop_size=20,
    pop_range=None, debug=False, sense="min",
)

model.alias                     # the function's __name__
model.tracker                   # [] unless solve(debug=True)
```

Each `solve` builds a fresh run (new buffer, new RNG, new tracker). Calling
`solve` again on the same model rebinds the seed/fields only if you call the
model again; the field values persist.

```python
model = my_algorithm(x=1)                 # one configuration...
a = model.solve(f, bounds, 100, 20)       # ...run against many objectives
b = model.solve(g, bounds, 200, 40)
```

## Kernels and the function style

The function style has **no compile-option channel**: there is no
`parallel`/`fastmath`/`cache` to pass at binding time. Numba kernels used by
a `@cm.declare` model are declared at module level and compile with their
decorator's own options — see
[Numba kernels](../compile/index.md) and
[Compilation options](../compile/options.md).

## Limitations at a glance

- fields: `cm.Argument` only (no `Constant`, no `Variable`);
- state: locals/closures, no per-instance attributes;
- no `prepare()` / compile options;
- `ctx.epoch` is fixed at 0.

If you need any of these, use the [class style](template.md).

## Full example

`example/hello_world_v1.py` is the complete, runnable version of the algorithm
above.
