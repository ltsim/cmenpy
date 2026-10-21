# Numba kernels

`cmenpy.compile` exposes three decorators. Each returns a `_Kernel` descriptor that
**compiles lazily on first use** (or eagerly via `cm.prepare(model)` /
`solve(..., prepare=True)`) and auto-casts its arguments. Population views
(`pop.X` / `pop.F`) can be passed straight to a kernel — they are unwrapped
internally, so no implicit `np.asarray` cast happens. `nopython=True` is always
enabled.

## Declaring specs

Specs can be passed **positionally** (`cm.Array[...]` / `cm.Scalar[...]` markers:
bare means untyped, e.g. `cm.Array[float, 2]` is float64 with rank 2) or as
**parameter annotations**:

```python
@cm.precompile
def hunt(X: "Array[float, 2]", a: "Scalar[float]"): ...

# mypy-clean form (identical meaning):
def hunt(X: typing.Annotated[npt.NDArray[np.float64], cm.Array[float, 2]]): ...
```

Plain (non-marker) hints are ignored; positional markers win over annotations.

## `@cm.precompile` (njit)

```python
@cm.precompile(parallel=True)
def hunt(X: "Array[float, 2]", leaders: "Array[float, 2]"):
    out = np.empty_like(X)
    for i in nb.prange(X.shape[0]):
        ...
    return out
```

Maps to `numba.njit`. `fastmath` and `parallel` are first-class options
(`parallel` also needs `nb.prange` in the body); any other numba option
(`cache=True`, `boundscheck=False`, …) is forwarded as a keyword.

!!! warning "Deprecated / removed"
    `cm.jit` remains as a **deprecated alias** that emits a `DeprecationWarning`.
    The old `@cm.parallel` is gone — use `@cm.precompile(parallel=True)`.

## `@cm.vectorize` (ufunc)

```python
@cm.vectorize
def scale(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]":
    return a * (2.0 * r - 1.0)
```

Elementwise ufunc with automatic casting. A fully typed kernel needs a full
signature: positionally `cm.Scalar[float, [cm.Scalar[float], cm.Scalar[float]]]`
(return dtype first), or a typed return annotation. A partially typed
`@cm.vectorize` raises `TypeError`. Accepts `fastmath`/`parallel`
(`parallel` maps to numba's `target="parallel"`).

## `@cm.reduce` (generalized ufunc)

The preferred form infers the layout from typed annotations — the **last
parameter is the output**:

```python
@cm.reduce
def matmul(a: "Array[float, 'm,n']", b: "Array[float, 'n,k']",
           out: "Array[float, 'm,k']"):
    ...
# layout (m,n),(n,k)->(m,k), outputs allocated by numba
```

The explicit `cm.Contract[[inputs...], output]` marker remains for full control:

```python
@cm.reduce(cm.Contract[[float, float, ..., ..., ...], float]])
def combine(a, b, s1, s2, s3, out): ...   # (n),(n),(),(),()->(n)
```

Contract shorthand: a plain dtype is `Array[dtype, "n"]`, `...` is a default-float
scalar, `Array[dtype]` is a whole-array argument with a private dimension, and
`Array[dtype, "m,n"]` names its dimensions. The legacy `(signatures, layout)` pair
is still accepted. Accepts `fastmath=True` and `parallel=True`
(`target="parallel"`).

## Model compile options

A `@cm.template` model sets options at instantiation and they propagate to every
kernel — `fastmath`/`cache` apply when the model *or* the kernel asks, while
`parallel` on a `precompile` kernel additionally requires the kernel to have opted
in with `parallel=True` (it needs `nb.prange`):

```python
model = GreyWolfOptimizer(seed=1, parallel=True, fastmath=True, cache=True)
model.prepare()              # force compilation, report signatures
cm.prepare(SomeClass)        # same for a standalone kernel holder
```

## API

Autodoc for the compile decorators, markers and introspection helpers lives in
the [API Reference → Compile](api/compile.md).
