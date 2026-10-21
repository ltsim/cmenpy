# Vectorized targets

Wrapping an objective with `cm.vectorized_target` declares that it can score
a whole `(pop, ndim)` block in one call. `KernelBuffer.apply` then dispatches
the entire active block through a single `evaluate_matrix` instead of one
`apply_along_axis` call per individual.

```python
import numpy as np

import cmenpy as cm


def sphere_block(x):            # x: (pop, ndim) -> (pop,)
    return np.sum(x**2, axis=1)


model = MyAlgorithm(seed=1)
best = model.solve(cm.vectorized_target(sphere_block), bounds, epochs, pop_size)
```

## The contract

- **In:** a 2-D array of shape `(k, ndim)` — the active solutions.
- **Out:** a 1-D array of shape `(k,)` — one fitness per row, in order.
- The block `x` *is* the live buffer's solution rows: do not modify it.
- The result must be assignable to the fitness column — same length, numeric.

Fitness equivalence with the row-by-row form is the whole point:
`block(x)[i] == row(x[i])` for every row.

## How the marker is detected

`TargetFunction` checks `getattr(target, "vectorized", False) is True`. The
identity comparison means:

- `cm.vectorized_target(f)` — always works (the wrapper sets
  `vectorized = True`);
- a class with a `vectorized = True` attribute and a `__call__` scoring a
  block — also accepted;
- a truthy stand-in (`vectorized = 1`, a non-empty string) — **not** accepted;
- a plain function — never vectorized.

`cm.vectorized_target` is the supported, self-documenting way to set it.
Calling `evaluate_matrix` on a non-vectorized target raises:

```
TypeError: Target is not vectorized; wrap it with `vectorized_target`.
```

## Recipes

### Pure NumPy

```python
def rastrigin_block(x):
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def rosenbrock_block(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2,
                  axis=1)
```

### Compiled block kernel

Combine a [`@cm.precompile`](../compile/precompile.md) kernel with the
wrapper — the kernel scores the block in Numba, the wrapper marks it:

```python
@cm.precompile
def sphere_block(X: "Array[float, 2]"):
    out = np.empty(X.shape[0])

    for i in range(X.shape[0]):
        acc = 0.0
        for j in range(X.shape[1]):
            acc += X[i, j] * X[i, j]
        out[i] = acc

    return out


model.solve(cm.vectorized_target(sphere_block), bounds, epochs, pop_size)
```

Inside a `@cm.template` model the same kernel is a method and inherits the
model's compile options.

## Pitfalls

- **Wrong axis** — `np.sum(x**2)` sums the whole block into a scalar; use
  `axis=1`.
- **Returning a scalar or a list** — must be a `(k,)` array; a scalar
  broadcasts and corrupts the fitness column.
- **Modifying `x`** — it is the population's own storage.
- **dtype drift** — return floats; integer outputs truncate fitness.
- **Shape mismatch** — returning `(k, 1)` or a transposed `(ndim, k)` fails
  (or silently broadcasts) when written to the fitness column.

## Not `@cm.vectorize`

[`@cm.vectorize`](../compile/vectorize.md) builds a NumPy *ufunc* from a
scalar kernel — it maps one value to one value and cannot turn `(pop, ndim)`
into `(pop,)`. For block objectives use `@cm.precompile` (or plain NumPy) and
mark the result with `cm.vectorized_target`.
