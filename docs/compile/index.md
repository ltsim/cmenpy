# Numba kernels

`cmenpy` wraps three Numba compiler paths — `njit`, `vectorize` and
`guvectorize` — in decorators that add three things Numba does not give you
on its own:

1. **Spec-driven casting.** Declared specs (`cm.Array`, `cm.Scalar`) coerce
   every argument to the declared dtype/layout before the compiled code sees
   it, so plain Python lists and misaligned arrays just work.
2. **Deferred compilation.** Each decorator returns a `_Kernel` descriptor
   that compiles on first use (or eagerly, when the signature is fully typed),
   so a model can inject its compile options before anything is compiled.
3. **Model-level compile options.** A `@cm.template` model can set
   `parallel`/`fastmath`/`cache` once at instantiation and every kernel it
   inherits — including ones mixed in from an Operator Module — picks them up.

`nopython=True` is always enabled.

## The three families

| Decorator | Compiles to | Kernel shape | Typical use |
| --------- | ----------- | ------------ | ----------- |
| [`@cm.precompile`](precompile.md) | `numba.njit` | any function: loops, branches, whole blocks | mutation, selection, block objectives |
| [`@cm.vectorize`](vectorize.md) | `numba.vectorize` (ufunc) | one scalar in, one scalar out | elementwise helpers: clip, scale, activation |
| [`@cm.reduce`](reduce.md) | `numba.guvectorize` (gufunc) | loop over core dimensions, output allocated | row/matrix kernels: row clip, matmul, outer |

Choosing one:

- **Loop over rows or a whole block yourself** → `@cm.precompile`.
- **One element at a time, no loops** → `@cm.vectorize`.
- **Loop over named core dimensions, with the output allocated for you** →
  `@cm.reduce`.

`cm.jit` still exists as a **deprecated alias** of `@cm.precompile` and emits
a `DeprecationWarning`.

## Where kernels live

A kernel can be declared in three places, and the difference matters only for
compile options:

```python
# 1. Module level — compiles with the decorator's own options.
@cm.precompile
def score(X: "Array[float, 2]"): ...

# 2. A @cm.template method — picks up the model's compile options.
@cm.template
class MyModel:
    @cm.precompile(parallel=True)
    def mutate(self, K: "Array[float, 2]"): ...

# 3. A @cm.module mixin — same as (2), through inheritance.
```

Only styles 2 and 3 receive a model's `parallel`/`fastmath`/`cache` flags;
a module-level kernel uses the options you passed to the decorator and nothing
else. See [Compilation options](options.md) for the exact precedence.

## Compilation lifecycle

Each decorator returns a `_Kernel` descriptor rather than a compiled
function:

- **Fully typed** signature (dtype *and* rank declared, or a raw numba
  signature string) → compilation happens at decoration time.
- **Partially typed or bare** (e.g. just `cm.Array`) → the descriptor stays
  lazy and compiles on first call, when numba infers the concrete types.
- Every distinct resolved option set (`fastmath`, `cache`, `parallel`, …)
  compiles once and is cached; switching a model's flags compiles a second
  variant rather than recompiling from scratch.

To force compilation before the hot loop, in order of preference:

```python
model = MyModel(seed=1, parallel=True)
model.prepare()              # report {kernel: [signatures]} after compiling
cm.prepare(model)            # same, works for a model instance or a class
model.solve(f, bounds, epochs, pop_size, prepare=True)   # compile inside solve
```

## API

Autodoc for the decorators, markers and introspection helpers lives in the
[API Reference → Functions](../api/functions.md#kernel-decorators).
