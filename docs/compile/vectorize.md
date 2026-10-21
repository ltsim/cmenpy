# `@cm.vectorize` (ufunc)

`@cm.vectorize` maps to `numba.vectorize`. It turns a **scalar** kernel —
one value in, one value out — into a NumPy ufunc that broadcasts over
arrays.

```python
import numpy as np

import cmenpy as cm


@cm.vectorize
def double(x: "Scalar[float]") -> "Scalar[float]":
    return 2.0 * x


double(np.array([1.0, 2.0]))        # -> array([2., 4.])
```

The decorated kernel **is** a `np.ufunc`: it broadcasts, accepts scalars, and
can be passed anywhere NumPy expects one.

## Signatures: the full-signature rule

A ufunc must know its return dtype before it is compiled. `@cm.vectorize`
accepts a full eager signature in four ways:

```python
# 1. Positional _Signature — return dtype first, then the argument list.
@cm.vectorize(cm.Scalar[float, [cm.Scalar[float], cm.Scalar[float]]])
def affine(r, a):
    return a * r + 1.0

# 2. Typed return annotation (all params typed too).
@cm.vectorize
def affine(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]":
    return a * r + 1.0

# 3. Raw numba signature string.
@cm.vectorize("float64(float64, float64)")
def add(x, y):
    return x + y

# 4. Bare / partially typed — deferred; numba infers at the first call.
@cm.vectorize
def affine(r: "Array", a: "Scalar"):
    return a * r + 1.0
```

!!! failure "Partially typed *and* precompiling raises"
    A kernel that has some typed specs but no typed return cannot be eagerly
    compiled:

    ```python
    @cm.vectorize
    def bad(x: "Scalar[float]"):     # TypeError: needs a typed return
        return x
    ```

    Either annotate the return (`-> "Scalar[float]"`), give the full
    positional signature, or drop the annotations and let it stay lazy.

A raw numba signature string wins over any annotation.

## Options

| Option | Effect |
| ------ | ------ |
| `fastmath=True` | Numba `fastmath` |
| `parallel=True` | `target="parallel"` — NumPy parallel ufunc; no `prange` needed |
| any other kwarg | forwarded verbatim (`cache=True`, `nopython=True`, …) |

`nopython` is forced on. For a model method, `parallel`/`fastmath`/`cache`
can also come from the model — see [Compilation options](options.md).

## Recipes

### Clip to bounds

The canonical use inside an optimizer body — no loops, pure scalar math:

```python
@cm.vectorize
def clip(x: "Scalar[float]", lo: "Scalar[float]", hi: "Scalar[float]") \
        -> "Scalar[float]":
    return min(max(x, lo), hi)


pop.compute << clip(~pop.extract, ctx.bounds.low, ctx.bounds.up)
```

### Affine scaling

```python
@cm.vectorize
def scale(x: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]":
    return a * x
```

### Gaussian perturbation of one coordinate

```python
@cm.vectorize
def perturb(x: "Scalar[float]", noise: "Scalar[float]",
            s: "Scalar[float]") -> "Scalar[float]":
    return x + s * noise
```

### Activation-style helper

```python
@cm.vectorize(fastmath=True)
def sigmoid(x: "Scalar[float]") -> "Scalar[float]":
    return 1.0 / (1.0 + np.exp(-x))
```

## `@cm.vectorize` vs `@cm.precompile` for objectives

Do **not** use `@cm.vectorize` for a whole-block objective. A ufunc maps one
scalar to one scalar — it cannot turn `(pop, ndim)` into `(pop,)`. For an
objective that scores a block, use
[`@cm.precompile`](precompile.md#whole-block-objective) plus
[`cm.vectorized_target`](../target/vectorized.md).

## Method vs standalone

Identical to `precompile`: a `@cm.template` method receives the model's
compile options at instantiation, a module-level kernel uses the decorator's.
Introspection is the same — `model.prepare()`, `cm.kernels`, `.signatures()`.

```python
model = MyModel(seed=1, parallel=True)
print(model.prepare())        # {"clip": ["float64(float64, float64, float64)"]}
```
