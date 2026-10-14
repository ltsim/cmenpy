# Target functions

A *target* is the objective you are optimizing. `cmenpy` wraps it in a
`TargetFunction` that counts function evaluations (NFE) and enforces an optional
budget, then scores the population every epoch.

## Evaluation modes

There are two ways a target is scored, and the difference is the single most
important performance lever in `cmenpy`:

| Mode | How it scores | When |
| ---- | ------------- | ---- |
| **Row-by-row** (default) | `np.apply_along_axis` — one Python call per individual | A plain callable |
| **Vectorized** (fast) | A **single** call over the whole `(pop, ndim)` block | Wrapped with `cm.vectorized_target` |

A plain Python callable works, but it pays a per-individual Python-call
overhead:

```python
def sphere(x):                     # x: one solution vector -> one float
    return float(np.sum(x**2))

model.solve(sphere, bounds, epochs, pop_size)     # scored row-by-row
```

A **vectorized target** receives the entire active block at once and returns one
fitness per row, so the whole population is dispatched in one call:

```python
import numpy as np

def sphere_block(X):               # X: (pop, ndim) -> (pop,)
    return np.sum(X**2, axis=1)

model.solve(cm.vectorized_target(sphere_block), bounds, epochs, pop_size)
```

## `cm.vectorized_target`

`cm.vectorized_target(fn)` marks `fn` as able to score a whole block, returning
a `VectorizedTarget`. The contract is strict:

- **Input** is a 2-D array of shape `(pop, ndim)`.
- **Output** is a 1-D array of length `pop` — one value per row.
- The result is identical to row-by-row evaluation, only dispatched once.

The marker is checked with `is True`, so it must be exactly `True` on the
wrapper. A truthy stand-in (e.g. `vectorized = "yes"`) does **not** count and the
target silently falls back to row-by-row. `cm.vectorized_target` is the only
supported way to set it.

!!! tip "When to reach for it"
    Any objective with per-solution math (sums, products, norms, trig) vectorizes
    cleanly with NumPy broadcasting. If your objective is already written as
    `(pop, ndim) -> (pop,)`, wrapping it is a near-free speedup — especially for
    large populations or expensive functions.

## Vectorize with Numba

For objectives heavy enough that NumPy still leaves time on the table, compile
the block kernel with [`@cm.precompile`](compile.md) (numba `njit`) and *then*
wrap it in `cm.vectorized_target`. Two things must line up:

1. The kernel takes the **whole block** — annotate it as `Array[float, 2]` and
   return a `(pop,)` array (e.g. with `np.sum(..., axis=1)`).
2. Wrap the compiled kernel in `cm.vectorized_target` so `cmenpy` scores the
   block in one call instead of falling back to row-by-row.

```python
import cmenpy as cm


@cm.precompile
def sphere_numba(X: "Array[float, 2]"):
    return np.sum(X**2, axis=1)          # (pop, ndim) -> (pop,)


best = model.solve(
    cm.vectorized_target(sphere_numba), bounds, epochs, pop_size
)
```

The kernel compiles on first evaluation and then reports its signature; wrap it
in `cm.vectorized_target` so `cmenpy` dispatches the block in one call instead of
falling back to row-by-row. The two pieces are complementary:

```text
@cm.precompile          -> compile the block math with numba
cm.vectorized_target   -> tell cmenpy to dispatch the block in one call
```

!!! note "`cm.prepare` is for models/classes"
    To force-compile the kernels declared *on a model class*, use
    `model.prepare()` / `cm.prepare(SomeClass)` (see
    [Numba kernels](compile.md#model-compile-options)). A standalone kernel like
    the one above instead compiles on first use and reports via `.signatures()`.

!!! note "Elementwise ufuncs"
    `@cm.vectorize` (a numba `ufunc`) is the tool for elementwise helpers, not
    whole-block objectives — it maps a scalar kernel over an array. For an
    objective you want the *block* form (`(pop, ndim) -> (pop,)`), prefer
    `@cm.precompile` as above. See [Numba kernels](compile.md) for both.

## NFE budget

`TargetFunction` counts every evaluation and can stop the run early. Pass
`max_nfe` to the wrapper; when the budget is spent it raises `MaxNumberOfCalls`:

```python
from cmenpy.target import MaxNumberOfCalls, TargetFunction

tf = TargetFunction(sphere, bounds, max_nfe=10_000)
try:
    model.solve(tf, bounds, epochs, pop_size)
except MaxNumberOfCalls:
    ...
```

The running count is available as `tf.nfe`, and the sense-aware dispatch lives in
`KernelBuffer.apply` (see [Kernel](api/kernel.md)).

## Choosing a target

- **Prototyping** — a plain callable is fine; start here.
- **Large populations / costly objectives** — wrap a NumPy `(pop, ndim) -> (pop,)`
  function with `cm.vectorized_target`.
- **Hot inner loop** — compile the block kernel with `@cm.precompile` and wrap it
  with `cm.vectorized_target`.
- **Strict evaluation budget** — set `max_nfe` and catch `MaxNumberOfCalls`.

Runnable examples of each pattern are in `example/` (see
[Examples](examples.md)).

## API

Autodoc for targets lives in the [API Reference → Target](api/target.md).
