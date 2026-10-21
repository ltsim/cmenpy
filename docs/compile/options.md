# Compilation options

Two places can set compile options: the **decorator** (`@cm.precompile(parallel=True)`)
and, for a `@cm.template` model, the **instantiation** (`MyModel(seed=1, parallel=True)`).
This page documents how the two merge, when a kernel sees which, and how to
force and inspect compilation.

## The three model options

| Option | Meaning |
| ------ | ------- |
| `parallel` | enable parallel compilation (see per-decorator semantics) |
| `fastmath` | enable Numba's `fastmath` (faster, less precise floating point) |
| `cache` | forward Numba's on-disk cache, skipping compilation in later processes |

Any *other* option (`boundscheck`, `nogil`, `inline`, …) can only be passed to
the decorator; the model channel accepts exactly these three keys.

```python
@cm.template
class MyModel:
    @cm.precompile(parallel=True)
    def mutate(self, K: "Array[float, 2]"): ...


model = MyModel(seed=1, parallel=True, cache=True)   # all kernels it owns
```

## Merging rules

The decorator's options are the base; the model's config upgrades them:

- **`fastmath`** — enabled if *either* the model or the decorator asks for it.
- **`cache`** — same: model **or** decorator enables it.
- **`parallel`, `@cm.precompile` (njit)** — the kernel must opt in **and** the
  model must allow it. A kernel declared without `parallel=True` never
  compiles parallel, even when the model sets it.
- **`parallel`, `@cm.vectorize` / `@cm.reduce` (ufuncs)** — enabled if *either*
  the model or the decorator asks for it. Ufuncs are parallel-safe by default,
  so the model can switch them on wholesale.
- **`nopython=True`** is forced for the ufunc decorators regardless of any
  other setting.

### `parallel` precedence

| Kernel opt-in | Model `parallel` | `@cm.precompile` result | `@cm.vectorize` / `@cm.reduce` result |
| ------------- | ---------------- | ----------------------- | ------------------------------------- |
| `True`        | `True`           | `parallel=True`         | `target="parallel"` |
| `True`        | not set          | `parallel=True`¹        | `target="parallel"` |
| not set       | `True`           | serial                  | `target="parallel"` |
| not set       | not set          | serial                  | serial |

¹ When the kernel is used standalone (module level, or accessed on the class)
there is no model config, so the decorator flag alone decides.

## Where the model config applies

A `_Kernel` is a descriptor. Compilation resolves options **at first use**:

```python
model = MyModel(seed=1, fastmath=True)
model.mutate(...)        # instance access → sees fastmath=True
MyModel.mutate           # class access → decorator options only
```

- **Instance access** (`self.kernel(...)`, `model.kernel(...)`) passes the
  model's `_compile_options`.
- **Class access** (`MyModel.kernel`) and **module-level** kernels compile with
  the decorator's options only.

The same kernel compiled under two different configs exists side by side: the
descriptor keeps one compiled function per distinct resolved option set, keyed
by the merged options. Switching a model's flags therefore compiles a second
variant rather than recompiling the first.

## Eager vs lazy

| Signature state | When it compiles |
| --------------- | ---------------- |
| fully typed (dtype **and** rank, or a raw numba signature string) | at decoration time |
| partially typed / bare | on first call |

Since options are only known at instantiation, model flags can never change
*whether* a signature is eager — only *with which options* the (re)compilation
happens.

## Forcing compilation

Compilation costs time on the first call. Warm up before timing or before a
long run:

```python
model = MyModel(seed=1, parallel=True, cache=True)

model.prepare()                       # -> {"mutate": ["none(...)"], ...}
cm.prepare(model)                     # same, model instance or class
model.solve(f, bounds, epochs, pop_size, prepare=True)   # compile inside solve
```

`prepare()` returns a `{kernel_name: [signatures]}` mapping and resolves
kernels through the MRO, so an inherited Operator Module kernel is reported
too.

## Inspecting a kernel

```python
kernel = model.mutate                  # resolves + compiles for this model
kernel.signatures()                    # ["none(Array(float64, 2, C), ...)"]
kernel.targetoptions                   # the merged numba options dict
```

`model.compile_options` shows what the model will contribute; see the
[API Reference → Functions](../api/functions.md#kernel-decorators) for the
full surface.
