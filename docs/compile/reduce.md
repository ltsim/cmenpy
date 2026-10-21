# `@cm.reduce` (guvectorize)

`@cm.reduce` maps to `numba.guvectorize`: a **generalized ufunc** that loops
over named core dimensions and writes into an output array Numba allocates
for you. It is the tool for row-wise and matrix kernels where the loop over
the data is the whole point but the *layout* is regular.

```python
import numpy as np

import cmenpy as cm


@cm.reduce
def clip(row: "Array[float, 'n']",
         low: "Scalar[float]",
         up: "Scalar[float]",
         out: "Array[float, 'n']"):
    for i in range(row.shape[0]):
        out[i] = min(max(row[i], low), up)


clip(np.array([[-5.0, 0.5, 5.0], [10.0, -10.0, 1.0]]), -1.0, 1.0)
# -> array([[-1. ,  0.5,  1. ], [ 1. , -1. ,  1. ]])
```

The **last parameter is the output**: it is allocated by Numba and never
passed in by the caller.

## Three forms

| Form | When |
| ---- | ---- |
| **Typed annotations** (preferred) | layout inferred from dimension labels; last parameter is the output |
| **`cm.Contract`** | you want explicit control over core types and layout |
| **Legacy `(signatures, layout)`** | porting existing `guvectorize` code |

### 1. Annotation-inferred (preferred)

Every parameter must be annotated with a **fully typed** spec, and the
dimension labels define the layout:

```python
@cm.reduce
def matmul(a: "Array[float, 'm,n']",
           b: "Array[float, 'n,k']",
           out: "Array[float, 'm,k']"):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            acc = 0.0
            for k in range(a.shape[1]):
                acc += a[i, k] * b[k, j]
            out[i, j] = acc
```

Here `n` is shared between `a` and `b` (a core dimension that must match),
while `m` and `k` are free — Numba loops over them and allocates the result.

You can add an explicit `layout=` alongside annotations:

```python
@cm.reduce(layout="(n),()->(n)")
def scale(row: "Array[float, 1]", factor: "Scalar[float]",
          out: "Array[float, 1]"):
    ...
```

### 2. `cm.Contract`

`Contract[[inputs...], output]` encodes the same information without
annotations, with shorthands (see [specs](specs.md#contract-shorthand)):

```python
@cm.reduce(cm.Contract[[float, float, ..., ..., ...], float])
def combine(a, b, s1, s2, s3, out):
    for i in range(a.shape[0]):
        out[i] = a[i] + b[i] * (s1 + s2 + s3)
# layout: (n),(n),(),(),()->(n)
```

- A plain dtype (`float`) means `Array[dtype, "n"]` — a 1-D array.
- `...` means a scalar.
- `Array[dtype]` without a rank gets a **private** dimension (`k0`, …),
  shared with nothing.

### 3. Legacy `(signatures, layout)`

The classic Numba form, still accepted — and the way to declare **multiple
signatures**:

```python
@cm.reduce(
    [
        [cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]],
        [cm.Array[np.float32, 1], cm.Scalar[float], cm.Array[np.float32, 1]],
    ],
    "(n),()->(n)",
)
def scale(row, factor, out):
    for i in range(row.shape[0]):
        out[i] = row[i] * factor
```

## Layout rules

- Every parameter is **fully typed** — a bare `cm.Array` raises.
- Dimension labels are matched by name across all arguments, including the
  output.
- Every output dimension must appear in some input.
- The output cannot be a scalar — a gufunc always writes into an array.
- Scalars are annotated `cm.Scalar[dtype]` and appear as `()` in the layout.

## Options

| Option | Effect |
| ------ | ------ |
| `fastmath=True` | Numba `fastmath` |
| `parallel=True` | `target="parallel"` — parallel over the free dimensions |
| any other kwarg | forwarded verbatim (`cache=True`, …) |

For a model method, the flags can also come from the model
([Compilation options](options.md)).

## Choosing between `reduce`, `vectorize` and `precompile`

| You are writing… | Use |
| ---------------- | --- |
| scalar → scalar, broadcast | [`@cm.vectorize`](vectorize.md) |
| a loop over core dimensions with a matching output layout | `@cm.reduce` |
| anything irregular: branches over rows, multiple outputs, early exits | [`@cm.precompile`](precompile.md) |
| a whole-block objective `(pop, ndim) -> (pop,)` | `@cm.precompile` + [`cm.vectorized_target`](../target/vectorized.md) |

## Introspection

```python
model = MyModel(seed=1)
print(cm.prepare(model))       # {"scale": ["(float64,)->(float64)", ...]}
```

`@cm.reduce` kernels compile at decoration time when fully typed, so
signature errors surface at import. See
[Troubleshooting](troubleshooting.md) for the common `TypeError`s.
