# Numba kernels

`cmenpy.compile` exposes three decorators. Each returns a `_Kernel` descriptor that
**compiles lazily on first use** (or eagerly via `MyModel(prepare=True)` at
construction, `cm.prepare(model)` / `solve(..., prepare=True)`) and auto-casts its
arguments. `nopython=True` is always enabled.

Numba is an **optional backend**: kernels compile when numba is installed
(`pip install "cmenpy[numba]"`) and automatically run as pure NumPy/Python when
it is not. You can also control compilation per model or per kernel with the
`numba` option — see [Running without Numba](#running-without-numba).

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

## Argument casting

Arguments are cast automatically before reaching a kernel. A declared spec
(`cm.Array[...]` / `cm.Scalar[...]`) takes precedence and forces dtype/layout, but
specs are *optional*: even a completely bare kernel converts any argument that
exposes the NumPy array protocol into a plain ndarray. Buffer views such as
`pop.X`, `pop.F` or `~pop.X` can therefore be passed directly, and the result of
the call is a NumPy array ready for the `<<` write-back:

```python
@cm.precompile
def double(X):                      # no specs needed
    return X * 2.0

pop.X << self.double(pop.X)         # view in, ndarray out
```

Plain ndarrays, NumPy scalars, lists and tuples pass through untouched, so
scalar/tuple results and numba's list handling are unaffected.

## Model compile options

A `@cm.template` model sets options at instantiation and they propagate to every
kernel — `fastmath`/`cache` apply when the model *or* the kernel asks, while
`parallel` on a `precompile` kernel additionally requires the kernel to have opted
in with `parallel=True` (it needs `nb.prange`):

```python
model = GreyWolfOptimizer(
    seed=1, numba=True, parallel=True, fastmath=True, cache=True
)
model.prepare()              # force compilation, report signatures
cm.prepare(SomeClass)        # same for a standalone kernel holder
```

`prepare=True` compiles at construction time, so you never have to call
`prepare()` yourself:

```python
model = MyOptimizer(seed=1, numba=True, prepare=True)  # kernels compiled now
model = MyOptimizer(seed=1, numba=True)                # lazy; model.prepare()
```

`prepare=False` (the default) keeps lazy compilation; `model.prepare()` remains
available either way. With `numba=False` (or no numba installed) there is nothing
to compile, and the kernels simply report the `["<python>"]` marker.

## Running without Numba

The `numba` option controls **whether compilation takes place at all**. It can be
set on the model — `MyOptimizer(numba=False)` — or on a single kernel —
`@cm.precompile(numba=False)` — and applies to all three decorators. Leaving it
out resolves automatically: kernels compile if numba is installed, and run
uncompiled if it is not. Asking for compilation explicitly (`numba=True`) when
numba is missing raises an `ImportError` with an install hint.

Uncompiled kernels keep their specs (so arguments are still auto-cast) and run
through pure-Python equivalents:

| Decorator | Uncompiled flavor |
| --------- | ----------------- |
| `@cm.precompile` | the plain function |
| `@cm.vectorize` | `numpy.vectorize` (elementwise) |
| `@cm.reduce` | a layout-driven NumPy loop over the core signature |

`prepare()` / `cm.prepare(model)` report uncompiled kernels as `["<python>"]`.
Kernels written with `from cmenpy import prange` work in both modes: the name
binds numba's `prange` when the backend is installed and the builtin `range`
otherwise.

```python
model = MyOptimizer(seed=1, numba=False)   # never compiles, numba or not
model.prepare()                             # {"hunt": ["<python>"], ...}
fitness, best = model.solve(f, bounds, 150, 15)  # same results, no JIT
```

## API

Autodoc for the compile decorators, markers and introspection helpers lives in
the [API Reference → Compile](api/compile.md).
