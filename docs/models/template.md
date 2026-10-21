# Class style (`@cm.template`)

`@cm.template` turns a class into a model. The class supplies
`initialize(self, ctx)` and `evolve(self, e, ctx)`; the model handles
validation, the epoch loop, tracking and kernel compilation.

```python
import numpy as np

import cmenpy as cm


@cm.template
class MyAlgorithm:
    a: cm.Argument[int, (1, 100)]                 # tunable, range-checked

    w: cm.Variable[np.ndarray]                    # internal state (starts None)

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                                       (pop.size, ctx.bounds.ndim))
        self.w = np.zeros(ctx.bounds.ndim)

    def evolve(self, e, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << np.clip(
            ~pop.extract + ctx.rng.uniform(-1, 1, (pop.size, ctx.bounds.ndim)),
            ctx.bounds.low, ctx.bounds.up,
        )


model = MyAlgorithm(a=3, seed=1)                  # instantiate, then:
best = model.solve(f, bounds, epochs=100, pop_size=20)
```

`@cm.template` returns a `TemplateOptimizerModel` wrapping the class. The
model is **not** instantiated until you call it.

## Class shape

| Member | Required | Role |
| ------ | -------- | ---- |
| `initialize(self, ctx)` | no (checked with `hasattr`) | one-off setup, runs once per `solve` |
| `evolve(self, e, ctx)` | yes | one epoch of the algorithm |
| fields | — | `cm.Argument` / `cm.Variable` / `cm.Constant` ([fields](fields.md)) |
| kernels | — | `@cm.precompile` / `@cm.vectorize` / `@cm.reduce` methods |
| helpers | — | plain methods and [Operator modules](modules.md) |

## Instantiation

```python
model = MyAlgorithm(seed=1, a=3, parallel=True, cache=True)
```

The binding call is `model(seed=None, *args, **kwargs)`:

- **`seed` is the first positional**; keyword use is recommended.
- Remaining keywords are field values (plus compile options).
- Positional values after the seed bind to fields **in declaration order**
  (`MyAlgorithm(1, 4)` → `seed=1`, first field `=4`); constants are skipped,
  which makes positional binding fragile — prefer keywords.

The generated `__init__` then:

1. pops the compile options (`parallel`/`fastmath`/`cache`) — see
   [Compilation options](../compile/options.md);
2. reads the class's `Annotated` type hints and validates each field
   (type, range, required-vs-default);
3. sets the fields as instance attributes;
4. delegates to the class's own `__init__`, if it has one (compile options
   are not forwarded to it).

### Errors at instantiation

| Error | When |
| ----- | ---- |
| `TypeError: Missing required argument: 'x'` | a range-declared field with no default and no value |
| `TypeError: Field 'x' must be of type int` | type mismatch |
| `ValueError: Value 5 out of range for 'x'` | outside `(min, max)` |
| `TypeError: Constant field 'P' cannot be passed to the constructor; it is fixed at declaration.` | passing a `Constant` |

## Lifecycle of a run

```python
model.solve(f, bounds, epochs, pop_size,
            pop_range=None, debug=False, sense="min", prepare=False)
```

1. **`prepare=True`** → all kernels of the class compile up front with the
   model's options.
2. A context is built: buffer, bounds, seeded RNG, tracker.
3. **`initialize(ctx)`** runs once — seed the population here.
4. **`evolve(e, ctx)`** runs for `e` in `0 .. epochs-1`. A **fresh
   `Context`** is built for every epoch, so `ctx.epoch` tracks the current
   epoch.
5. The best row of the buffer is returned as `(best_fitness, best_x)`.

Because state lives on `self`, a second `solve` on the same model **reuses
the instance**: `initialize` runs again, but stale `self` attributes persist
unless you reset them there.

## Kernels as methods

```python
@cm.template
class MyAlgorithm:
    @cm.precompile(parallel=True)
    def mutate(self, K: "Array[float, 2]", s: "Scalar[float]"):
        out = np.empty_like(K)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                out[i, j] = K[i, j] + s * (2.0 * (i % 3) - 1.0)
        return out
```

Called through the instance (`self.mutate(...)`, `model.mutate(...)`), the
kernel inherits the model's compile options; accessed on the class
(`MyAlgorithm.mutate`) it uses the decorator's. Operator-module kernels mixed
into the class behave identically.

## Introspection and warm-up

```python
model.alias                     # "MyAlgorithm"
model.model_class               # the wrapped class
model.compile_options           # {"parallel": True, "fastmath": False, ...}
model.prepare()                 # force compilation, {kernel: [signatures]}
model.tracker                   # [] unless solve(debug=True)
```

`solve(..., prepare=True)` compiles inside the run — the right default before
benchmarking or production runs.

## Inheritance

Plain base classes work as usual (fields, methods and kernels are
inherited). For cross-cutting operators shared by several models, prefer
[`@cm.module`](modules.md) Operator Modules.

## Full example

`example/hello_world_v2.py` is a runnable model in this style; the
[Build a custom model](../tutorial-custom-model.md) tutorial implements a
complete (μ + λ) Evolution Strategy step by step.
