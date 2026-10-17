# Models

A *model* is an optimizer you describe once and then run against any objective.
`cmenpy` offers two declaration styles that produce the same contract — a
callable exposing:

```python
(best_fitness, best_x) = model.solve(
    f, bounds, epochs, pop_size, pop_range=None, debug=False, sense="min"
)
```

- **`@cm.declare`** wraps a *function*. Best when one function body drives the
  whole run and you want parameters bound by keyword.
- **`@cm.template`** wraps a *class*. Best when you want per-instance state,
  an explicit `initialize`/`evolve` split, and numba kernels that inherit
  compile options.

Both yield a `BaseOptimizer` with `.solve(...)`, `.alias` and `.tracker`.

## `@cm.declare` over a function

```python
@cm.declare(x=cm.Argument[int, (0, 3), 2])
def algo(args, epoch, ctx):
    pop = cm.PopulationSwarm(ctx.buff)
    pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                                   (pop.size, ctx.bounds.ndim))
    for _ in epoch:
        ...  # one epoch of your operator
```

The decorated function receives three arguments:

| Argument | Meaning |
| -------- | ------- |
| `args` | The bound `cm.Argument` fields (read `args.x`) |
| `epoch` | An `EpochIteration` — iterate it, or read `epoch.max` |
| `ctx`   | The run context (see [below](#the-ctx-object)) |

Parameters are declared as keyword `cm.Argument` fields and **bound by calling
the model first**, which returns a ready-to-run optimizer:

```python
model = algo(x=1)            # bind the declared field
best  = model.solve(f, bounds, epochs=150, pop_size=15)
```

Because the whole run lives inside one function, the loop is explicit: you
iterate `epoch` yourself. This style mirrors a plain script and is the easiest
starting point — see `example/hello_world_v1.py`.

## `@cm.template` over a class

```python
@cm.template
class MyAlgorithm:
    a: cm.Argument[int, (1, 100)]   # tunable, range-checked
    w: cm.Variable[np.ndarray]      # free internal state

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(-1, 1, (pop.size, ctx.bounds.ndim))

    def evolve(self, e, ctx) -> None:
        ...  # advance one epoch; called once per epoch
```

The lifecycle is fixed and driven by `solve`:

- **`initialize(ctx)`** runs once, before any epoch — seed the population and
  set up state here.
- **`evolve(e, ctx)`** runs once per epoch, with `e` the current epoch index.

Fields are declared as class annotations:

| Marker | Meaning |
| ------ | ------- |
| `cm.Argument[type]` | Required, no range/default |
| `cm.Argument[type, default]` | With a default value |
| `cm.Argument[type, (min, max)]` | Range-checked, required |
| `cm.Argument[type, (min, max), default]` | Range-checked, with default |
| `cm.Variable[type]` | Free internal state (no range, no default) |
| `cm.Constant[type, value]` | Fixed, read-only — never constructor-bound |

`cm.Constant` is checked at declaration and exposed through the class, so
`self.name` works but `MyAlgorithm(name=...)` raises `TypeError`.

Bind fields (and the seed) via the constructor call, then `solve`:

```python
model = MyAlgorithm(a=3, seed=1)
best  = model.solve(f, bounds, 150, 15)
```

The template style is the natural home for numba kernels: declare them as
methods with `@cm.precompile` / `@cm.vectorize` / `@cm.reduce`, and they
receive the model's compile options at instantiation —
`MyAlgorithm(seed=1, parallel=True, fastmath=True, cache=True)` (see
[Numba kernels](compile.md#model-compile-options)). See
`example/hello_world_v2.py` and the optimizers in `example/`.

!!! tip "Choosing a style"
    Reach for `@cm.declare` for quick, script-like experiments and
    `@cm.template` for reusable optimizers with state, compiled kernels, or an
    `initialize`/`evolve` split. For a full walk-through, see
    [Build a custom model](tutorial-custom-model.md).

## The `ctx` object

The context passed to `initialize`/`evolve` exposes:

| Attribute | Meaning |
| --------- | ------- |
| `ctx.buff` | The `KernelBuffer` backing the population |
| `ctx.bounds` | The search-space `Bounds` (`low`, `up`, `ndim`) |
| `ctx.rng` | The random generator |
| `ctx.sense` | `"min"` or `"max"` — propagate it instead of hard-coding |
| `ctx.epoch` / `ctx.epochs` | Current epoch / total |
| `ctx.target` / `ctx.f` | The wrapped objective |

!!! note
    Use `ctx.sense` (or the `cm.is_best` / `cm.is_worst` comparators, which take
    a sense) rather than hard-coding `<` or `>`, so your model works for both
    minimization and maximization.

## Introspection

A template model exposes a few useful hooks:

- `model.model_class` — the wrapped class.
- `model.compile_options` — the resolved `parallel`/`fastmath`/`cache` flags.
- `model.prepare()` — force-compile all kernels and return their signatures.
- `model.tracker` — the per-epoch history (populated when `solve(..., debug=True)`).

## Operator modules (`@cm.module`)

`@cm.template` models can share behavior through **Operator Modules**. A class
decorated with `@cm.module` is a reusable bundle — compiled kernels, plain helper
methods and `cm.Argument` / `cm.Variable` / `cm.Constant` fields — that any
number of models inherit (the Ruby-`module` analogue in this framework; note it
returns a class, not a Python module):

```python
@cm.module
class MutationOps:
    scale: cm.Constant[float, 0.3]

    @cm.vectorize
    def clip(x: cm.Array, lo: cm.Scalar, hi: cm.Scalar):
        return min(max(x, lo), hi)

@cm.template
class MyOptimizer(MutationOps):
    def initialize(self, ctx) -> None: ...
    def evolve(self, e, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << self.clip(~pop.extract * self.scale, -1, 1)
```

- Mixin application is ordinary inheritance; collisions follow the MRO and the
  leftmost base wins.
- Kernels declared on an Operator Module receive the model's compile options at
  instantiation, exactly like kernels declared on the model itself, and
  `model.prepare()` reports them (it resolves through the MRO).
- `cm.Constant` fields declared on the Operator Module are inherited unchanged
  and stay read-only.

See `example/operator_module.py` for two models sharing one Operator Module.

## API

Autodoc for the model layer (decorators, markers, optimizers, protocols) lives
in the [API Reference → Models](api/models.md). To build your own model from
scratch, follow [Build a custom model](tutorial-custom-model.md).
