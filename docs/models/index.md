# Models

A *model* is an optimizer you describe once and then run against any
objective. `cmenpy` offers two declaration styles that produce the same
contract — a callable exposing:

```python
(best_fitness, best_x) = model.solve(
    f, bounds, epochs, pop_size, pop_range=None, debug=False, sense="min"
)
```

- **`@cm.declare`** wraps a *function*: one body drives the whole run, and
  parameters are bound by keyword.
- **`@cm.template`** wraps a *class*: per-instance state, an explicit
  `initialize`/`evolve` split, fields, and numba kernels that inherit
  compile options.

Both produce a `BaseOptimizer` with `.solve(...)`, `.alias` and `.tracker`.

## Where and when to use which

| | `@cm.declare` (function) | `@cm.template` (class) |
| - | ------------------------ | ---------------------- |
| Run structure | one function body; you iterate the epochs yourself | `initialize` once, then `evolve` per epoch |
| Parameters | `cm.Argument` only, keyword-bound at call time | `cm.Argument`, `cm.Variable`, `cm.Constant` |
| Per-instance state | closure/local variables only | instance attributes (`self.w`) |
| Numba kernels | module-level (decorator options only) | methods that receive the model's compile options |
| Compile options | none | `parallel` / `fastmath` / `cache` at instantiation |
| Reuse/composition | copy/paste or plain functions | inheritance and [`@cm.module`](modules.md) Operator Modules |
| Introspection | `alias`, `tracker` | plus `model_class`, `compile_options`, `prepare()` |
| Best for | scripts, quick experiments, one-off algorithms | reusable optimizers, anything with state or kernels |

**Choose `@cm.declare` when** the algorithm is a single function, you want
parameters bound by keyword, and you have no per-instance state or compiled
kernels needing model options. It is the shortest path from idea to run.

**Choose `@cm.template` when** you want the `initialize`/`evolve` split, keep
state across epochs on `self`, declare constants, compile numba kernels that
should honor `parallel`/`fastmath`/`cache`, share operators through
[`@cm.module`](modules.md), or introspect/prepare the model.

The same algorithm in both styles:

```python
@cm.declare(x=cm.Argument[int, (0, 3), 2])
def random_search(args, epoch, ctx):
    pop = cm.PopulationSwarm(ctx.buff)
    pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                                   (pop.size, ctx.bounds.ndim))
    for _ in epoch:
        pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                                       (pop.size, ctx.bounds.ndim))


@cm.template
class RandomSearch:
    x: cm.Argument[int, (0, 3), 2]

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                                       (pop.size, ctx.bounds.ndim))

    def evolve(self, e, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                                       (pop.size, ctx.bounds.ndim))
```

## Lifecycle

**Function style** — one pass:

```
@cm.declare → FunctionOptimizerModel
model = algo(x=1)                       # bind fields (+ optional seed)
solve(...)  → build context
            → func(args=args, epoch=epoch_it, ctx=Context(...))   # once
            → you iterate `epoch`; ctx.epoch stays 0
            → return (best_fitness, best_x)
```

**Class style** — per-epoch hooks:

```
@cm.template → TemplateOptimizerModel (class wrapped, not yet instantiated)
model = MyAlgorithm(x=1, seed=1)        # instantiate: bind fields + options
solve(..., prepare=False) → [prepare(): compile kernels]
                          → initialize(ctx)                # once
                          → for e in epoch_it: evolve(e, ctx=Context(...))
                          → return (best_fitness, best_x)
```

A fresh `Context` is built for every hook call in the class style, so
`ctx.epoch` tracks the current epoch. In the function style the single
`Context` is built before your function runs — use the `epoch` iterator's
counter, not `ctx.epoch`, inside `@cm.declare`.

## `solve` parameters

| Parameter | Meaning |
| --------- | ------- |
| `f` | the objective; any callable, optionally wrapped with [`cm.vectorized_target`](../target/vectorized.md) |
| `bounds` | a `Bounds` or a bounds sequence; defines `ndim` |
| `epochs` | total epochs (iterated as `0 .. epochs-1`) |
| `pop_size` | initial population size |
| `pop_range` | `(min, max)` allowed sizes for dynamic populations |
| `debug` | record per-epoch history in `model.tracker` |
| `sense` | `"min"` (default) or `"max"` |
| `prepare` | *(template only)* compile the class's kernels before running |

Returns `(best_fitness, best_x)` — the best row of the buffer, respecting
`sense`.

!!! info "Planned for 2026b2"
    `solve` will accept two opt-in run-control arguments alongside these:
    `terminator=` for composable stopping criteria
    ([#33](https://github.com/ltsim/cmenpy/issues/33)) and `history=` for
    in-memory / on-disk run records
    ([#35](https://github.com/ltsim/cmenpy/issues/35)). Today the equivalents
    are the fixed `epochs` cap and `debug=True` with `model.tracker`.

## The `ctx` object

Every hook receives a read-only `Context`:

| Attribute | Meaning |
| --------- | ------- |
| `ctx.bounds` | the search space (`low`, `up`, `ndim`) |
| `ctx.buff` | the `KernelBuffer` backing the [population](../population/index.md) |
| `ctx.rng` | seeded random generator |
| `ctx.sense` | `"min"` or `"max"` |
| `ctx.epoch` / `ctx.epochs` | current epoch / total (template style) |
| `ctx.target` | the wrapped objective ([budget](../target/budget.md) accounting lives on the run's own wrapper) |

!!! note "Per-dimension bounds"
    `bounds.low` / `bounds.up` are the **first dimension's** scalars. For
    unequal per-dimension intervals pass the arrays straight to the sampler
    (e.g. an opfunu objective's `f.lb` / `f.ub` — see
    [Test functions](../benchmarks.md)). `Bounds.lows`, `ups`, `sample()` and
    `clip()` helpers land in 2026b2
    ([#37](https://github.com/ltsim/cmenpy/issues/37)).

## Where to go next

- [Function style (`declare`)](declare.md)
- [Class style (`template`)](template.md)
- [Fields and markers](fields.md) — `Argument`, `Variable`, `Constant`
- [Operator modules](modules.md) — `@cm.module` mixins
- [Build a custom model](../tutorial-custom-model.md) — full class-style tutorial
