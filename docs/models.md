# Models

Two declaration styles, both producing an optimizer with:

```python
(best_fitness, best_x) = model.solve(
    f, bounds, epochs, pop_size, pop_range=None, debug=False, sense="min"
)
```

## `@cm.declare` over a function

```python
@cm.declare(x=cm.Argument[int, (0, 3), 2])
def algo(args, epoch, ctx):
    ...
```

Parameters are bound by calling the model first: `algo(x=1).solve(...)`.

## `@cm.template` over a class

```python
@cm.template
class MyAlgorithm:
    a: cm.Argument[int, (1, 100)]   # tunable, range-checked, with default
    w: cm.Variable[np.array]        # free internal state

    def initialize(self, ctx) -> None: ...
    def evolve(self, e, ctx) -> None: ...
```

Fields use `cm.Argument[type, (min, max), default]` / `cm.Variable[type]`.
A model can set compile options at instantiation —
`Model(seed=1, parallel=True, fastmath=True, cache=True)` — which propagate to
its kernels (see [Numba kernels](compile.md#model-compile-options)). Useful
introspection: `model.model_class`, `model.compile_options`, `model.prepare()`,
`model.tracker`.

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

## API

::: cmenpy.declare
::: cmenpy.template
::: cmenpy.Argument
::: cmenpy.Variable
::: cmenpy.FunctionOptimizerModel
::: cmenpy.TemplateOptimizerModel
::: cmenpy.EpochIteration
