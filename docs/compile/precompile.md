# `@cm.precompile` (njit)

`@cm.precompile` maps to `numba.njit`. It is the general-purpose kernel: any
function body — loops, branches, whole-block math — compiled with `nopython`
always on.

```python
import numpy as np

import cmenpy as cm


@cm.precompile
def square(X: "Array[float, 2]"):
    return X**2
```

## Calling forms

All of these are accepted; combine freely.

```python
# Bare — lazy, inference at first call.
@cm.precompile
def f(X): ...

# Options.
@cm.precompile(parallel=True, fastmath=True)
def f(X): ...

# Positional specs (win over annotations).
@cm.precompile(cm.Array[float, 2], cm.Scalar[float])
def f(X, a): ...

# Annotations.
@cm.precompile
def f(X: "Array[float, 2]", a: "Scalar[float]"): ...

# Raw numba signature string.
@cm.precompile("float64(float64)")
def f(x): ...
```

## Options

| Option | Effect |
| ------ | ------ |
| `fastmath=True` | Numba `fastmath` — faster, less precise FP |
| `parallel=True` | Numba `parallel`; the body **must** use `nb.prange` |
| any other kwarg | forwarded to `njit` verbatim (`cache=True`, `boundscheck=False`, `nogil=True`, `inline="always"`, …) |

```python
import numba as nb
import numpy as np


@cm.precompile(cm.Array, cm.Scalar, parallel=True)
def scale_rows(A, factor):
    out = np.empty_like(A)

    for i in nb.prange(A.shape[0]):     # prange is required for parallel
        out[i] = A[i] * factor

    return out
```

!!! warning "`parallel` needs `prange`"
    `parallel=True` only parallelizes `numba.prange` loops. A kernel with no
    `prange` compiles fine but runs serially — see
    [Troubleshooting](troubleshooting.md#parallel-has-no-effect).

For a model method, `parallel` has a second gate: the kernel must opt in
**and** the model must enable it (see
[Compilation options](options.md#parallel-precedence)). At module level only
the decorator flag matters.

## Return conventions

- Return a new array (`np.empty`, `np.empty_like`) or a scalar.
- In-place kernels that write into a passed `out` array work too, as long as
  the layout matches the spec.
- The returned dtype is whatever the body computes; declare specs if you need
  a specific one.

## Recipes

### Whole-block objective

The pattern from [Target functions](../target/vectorized.md): score a whole
`(pop, ndim)` block in one call.

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
```

### Mutation with clipping

The kernel from [Build a custom model](../tutorial-custom-model.md): mutate a
block of offspring and clip to bounds, with scalars cast from `ctx.bounds`.

```python
@cm.precompile
def mutate(K: "Array[float, 2]", N: "Array[float, 2]",
           s: "Scalar[float]", lo: "Scalar[float]", hi: "Scalar[float]"):
    out = np.empty_like(K)

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            v = K[i, j] + s * N[i, j]
            out[i, j] = min(max(v, lo), hi)

    return out
```

### Pairwise distances

A row-per-thread reduction, the classic `prange` case.

```python
import numba as nb
import numpy as np


@cm.precompile(parallel=True)
def distances(X: "Array[float, 2]", Y: "Array[float, 2]"):
    D = np.empty((X.shape[0], Y.shape[0]))

    for i in nb.prange(X.shape[0]):
        for j in range(Y.shape[0]):
            acc = 0.0
            for k in range(X.shape[1]):
                d = X[i, k] - Y[j, k]
                acc += d * d
            D[i, j] = np.sqrt(acc)

    return D
```

## Method vs standalone

A kernel declared as a method of a `@cm.template` class is called on the
instance (`self.mutate(...)`), which resolves the model's compile options; a
module-level kernel is called directly. Both compile on first use, and both
can be inspected:

```python
model = MyModel(seed=1, parallel=True)
print(model.prepare())        # {"mutate": ["none(Array(float64, 2, C), ...)"]}
```

See [Compilation options](options.md) for what the model contributes and
[Troubleshooting](troubleshooting.md) when a kernel compiles with unexpected
options.

## Performance notes

- **First call pays the compile.** Warm up with `model.prepare()` or
  `solve(..., prepare=True)` before timing.
- **`cache=True`** (decorator kwarg or model flag) writes Numba's on-disk
  cache, skipping compilation on the next process.
- **Keep dtypes stable.** A `float32` argument into a float64 spec upcasts
  (one copy, one signature); a mix of dtypes compiles several signatures.
- Avoid Python objects (lists, dicts, strings) inside the body — Numba's
  `nopython` mode rejects them.
