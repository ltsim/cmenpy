# Typing and cast specs

Numba compiles on concrete types, not on Python type hints. The spec markers
tell `cmenpy` two things at once: **how to cast** each argument before the
call (dtype, memory layout) and **how to type** it inside Numba (dtype,
rank). Plain hints (`npt.NDArray[np.float64]`) are ignored — only markers
count.

## The markers

### `cm.Array` — array parameters

```python
cm.Array                      # untyped: np.asarray, any dtype/rank
cm.Array[float]               # float64, rank inferred as 1 by default in Contract
cm.Array[float, 2]            # float64, rank 2 (C-contiguous)
cm.Array[float, "m,n"]        # float64, rank 2, dimensions named
cm.Array[float, 2, "F"]       # float64, rank 2, Fortran-contiguous
```

| Subscript | Meaning |
| --------- | ------- |
| `dtype` | NumPy dtype the argument is coerced to |
| `ndim` | integer rank **or** a string of comma-separated dimension labels |
| `layout` | `"C"` (default), `"F"` or `"A"` (any) — the memory layout |

### `cm.Scalar` — scalar parameters

```python
cm.Scalar            # untyped: coerced with float()
cm.Scalar[float]     # float64 scalar
cm.Scalar[int]       # int64 scalar
```

### `cm.Contract` — full signatures for `@cm.reduce`

```python
cm.Contract[[float, float, ..., ..., ...], float]     # (n),(n),(),(),()->(n)
cm.Contract[[cm.Array[float, "m,n"], cm.Array[float, "n,k"]],
            cm.Array[float, "m,k"]]                   # (m,n),(n,k)->(m,k)
```

`Contract[[inputs...], output]` encodes both the core types and the layout;
`@cm.reduce` reads it directly. See [reduce](reduce.md) for the shorthand
rules.

## Where specs go

Specs can be passed **positionally** to the decorator or declared as
**parameter annotations**. All of these are equivalent:

```python
# 1. Positional specs (win over any annotation).
@cm.precompile(cm.Array[float, 2], cm.Scalar[float])
def hunt(X, a): ...

# 2. String annotations.
@cm.precompile
def hunt(X: "Array[float, 2]", a: "Scalar[float]"): ...

# 3. typing.Annotated with a marker (mypy/pyright-clean).
@cm.precompile
def hunt(X: typing.Annotated[npt.NDArray[np.float64], cm.Array[float, 2]],
         a: typing.Annotated[float, cm.Scalar[float]]): ...

# 4. typing.Annotated with a string marker.
@cm.precompile
def hunt(X: typing.Annotated[np.ndarray, "Array[float, 2]"]): ...
```

Rules:

- **Positional specs win** over annotations, and a raw numba signature string
  (e.g. `"float64(float64, float64)"`) wins over everything.
- A **plain type hint is not a spec** — it is ignored, and the kernel stays
  untyped (lazy).
- String annotations are resolved in the function's globals plus `Array` and
  `Scalar`, so `from __future__ import annotations` and stringified hints
  work.
- In annotations use the bare names `Array`/`Scalar` (they are injected into
  the namespace); positional specs use `cm.Array`/`cm.Scalar`.

## What casting does

Casting runs before the compiled call, on every argument, following its
spec:

| Spec | Scalar input | Array input |
| ---- | ------------ | ----------- |
| `cm.Array` (untyped) | `np.asarray(value)` | `np.asarray(value)` |
| `cm.Array[float]` | `np.ascontiguousarray(value, dtype=float64)` | same |
| `cm.Array[float, 2, "F"]` | `np.asfortranarray(value, dtype=float64)` | same |
| `cm.Scalar` | `float(value)` | `float(value)` |
| `cm.Scalar[float]` | `np.dtype(float64).type(value)` | `np.dtype(float64).type(value)` |

Consequences worth knowing:

- A Python `list` becomes an array (`cm.Array` kernels accept lists).
- A `float32` array passed to a `cm.Array[float, 2]` kernel is **upcast** —
  the float64 signature is the only one compiled.
- A Fortran-ordered array passed to a `"C"` kernel is copied into C order.
- Casting can allocate; in a hot inner loop pass already-typed arrays and
  avoid specs that force a copy.

## Eager vs lazy compilation

A spec is *typed* when it has both a dtype and a rank:

- **All specs typed** → the kernel compiles at decoration time.
- **Any spec untyped** (`cm.Array`, `cm.Array[float]` without rank) → the
  kernel stays lazy and compiles on first call, when numba sees the concrete
  arguments.

```python
@cm.precompile(cm.Array[float, 2])     # compiled at import
def total(A): ...

@cm.precompile(cm.Array)               # lazy: compiles on first call
def total(A): ...
```

The trade-off: eager kernels surface signature errors immediately and can be
inspected with `cm.prepare`; lazy kernels compile once per concrete argument
type they meet.

## Contract shorthand

Inside `cm.Contract`:

- A plain dtype (`float`, `np.float32`) means `Array[dtype, "n"]` — a 1-D
  array.
- `...` means a default-float **scalar**.
- `Array[dtype]` (no rank) gets a **private dimension** (`k0`, `k1`, …) — a
  per-argument axis that shares nothing with other arguments.
- Named dimensions (`"m,n"`) are shared by name across all arguments.
- Every output dimension must appear in some input, and the output cannot be
  a scalar.

```python
cm.Contract[[float, cm.Array[float]], float]     # (n),(k0)->(n): private k0
cm.Contract[[float, cm.Array[float, "m"]], cm.Array[float, "n,m"]]   # (n),(m)->(n,m)
```
