# Troubleshooting

Symptom → cause → fix, using the messages the decorators actually raise.

## Signature errors at decoration time

### `Precompiled @cm.vectorize needs a full signature`

A `@cm.vectorize` kernel was given positional specs with a dtype but no full
signature (the return type is missing). A ufunc must know its result dtype
before it compiles.

```python
@cm.vectorize(cm.Scalar[float])        # TypeError
def bad(r, a): ...

# fix — return dtype first, then the argument list:
@cm.vectorize(cm.Scalar[float, [cm.Scalar[float], cm.Scalar[float]]])
def good(r, a): ...
```

### `Precompiled @cm.vectorize with annotations needs a typed return`

Same cause, annotation form: parameters are typed but the return is not.

```python
@cm.vectorize
def bad(x: "Scalar[float]"): ...       # TypeError

@cm.vectorize
def good(x: "Scalar[float]") -> "Scalar[float]": ...
```

Either annotate the return, pass the full positional signature, or drop the
annotations and let the kernel stay lazy.

### `cm.reduce needs signatures or fully typed parameter annotations`

A `@cm.reduce` kernel has neither a signature form nor usable annotations.
`reduce` always needs a complete layout — give one of the three forms
(described in [reduce](reduce.md#three-forms)):

```python
@cm.reduce(layout="(n),()->(n)")            # annotations + layout
@cm.reduce(cm.Contract[[float, float], float])   # Contract
@cm.reduce([cm.Array[float, 1], cm.Scalar[float],
            cm.Array[float, 1]], "(n),()->(n)")   # legacy signatures
```

### `cm.reduce requires fully typed parameter annotations`

Some parameter is untyped or bare (`cm.Array` with no dtype). Every `reduce`
parameter — including the output — must carry dtype **and** dims:

```python
@cm.reduce
def bad(row: "Array[float]", out: "Array[float, 'n']"): ...   # TypeError

@cm.reduce
def good(row: "Array[float, 'n']", out: "Array[float, 'n']"): ...
```

### `cm.reduce requires fully typed specs, e.g. cm.Array[float, 1] or cm.Scalar[float]`

The legacy form got a bare marker. Add dtypes to every entry of the signature
list.

### `Layout '(n),()->(n)' expects 2 inputs, but 3 specs were given`

The layout string and the signature disagree on arity. Count: `(...)->(...)`
means every atom before `->` is an input (including scalars `()`), and the
atom after it is the output — the output is *also* one of the specs.

## `Contract` errors

| Message | Cause | Fix |
| ------- | ----- | --- |
| `Contract array specs need a dtype, e.g. Array[float].` | bare `Array` inside a Contract | `Array[float, "n"]` |
| `Contract scalar specs need a dtype, e.g. Scalar[float].` | bare `Scalar` | `Scalar[float]` |
| `Unsupported Contract spec: <value>.` | entry that is not a dtype, `...`, `Array[...]` or `Scalar[...]` | use one of the shorthands ([specs](specs.md#contract-shorthand)) |
| `cm.Contract does not support a scalar output; use an Array[...].` | output is a scalar | a gufunc always writes an array — `Array[...]` output |
| `Contract output dimensions ['m'] do not appear in any input.` | an output dimension label is not in any input | output dims must be a subset of input dims; fix the labels |
| `The first Contract argument must be a list of specs.` | `Contract[[...], out]` written without the inner list | keep the `[[inputs...], output]` shape |

## Runtime and typing errors

### Numba `TypingError: No implementation of function ... found`

The compiled body does not match the declared spec. Check in order:

- actual dtype vs declared (`float32` argument into a `float64` spec is cast,
  but a scalar spec applied to an array casts to a scalar — usually not what
  you meant);
- rank vs declared (`cm.Array[float, 2]` on a 1-D array fails inside the body);
- unsupported constructs in the body (Python objects, dynamic attributes) —
  Numba's `nopython` mode rejects them ([precompile](precompile.md)).

### `Target is not vectorized; wrap it with `vectorized_target`.`

Raised by `TargetFunction.evaluate_matrix` when the objective is not marked
vectorized. Wrap it with `cm.vectorized_target`, or give your callable a
`vectorized = True` attribute. See
[Vectorized targets](../target/vectorized.md).

## Options and performance

### Parallel has no effect

- **`@cm.precompile`:** the body must use `nb.prange`. A plain `range` loop
  compiles and runs serially.
- **Model method:** for `precompile` the kernel must also opt in — declaring
  the method with `@cm.precompile` and setting only `model=parallel=True` is
  not enough ([options](options.md#parallel-precedence)).
- **ufuncs** (`vectorize`/`reduce`) do not need `prange`; they use
  `target="parallel"`.

### A kernel compiles with the wrong options

Class-level or standalone access ignores the model's flags:

```python
MyModel.mutate            # decorator options only
model = MyModel(parallel=True)
model.mutate              # sees the model options
```

Call the kernel on an instance. Confirm with `kernel.targetoptions`.

### The first call is slow

That is the compilation. Warm up with `model.prepare()`,
`cm.prepare(model)`, or `solve(..., prepare=True)`, and set `cache=True`
(decorator or model flag) to reuse Numba's on-disk cache across processes.

### Several signatures compile unexpectedly

Each distinct resolved option set compiles its own variant, and each distinct
concrete argument type compiles its own signature. If you see extra
signatures, check for mixed dtypes reaching a cast spec, or a model and a
decorator disagreeing on `fastmath`/`cache` across calls. Inspect with
`kernel.signatures()`.

## Debugging checklist

```python
model = MyModel(seed=1, parallel=True)

print(model.prepare())          # {name: [signatures]} across the MRO
print(model.compile_options)    # what the model contributes

k = model.mutate                # resolve through the instance
print(k.signatures())           # compiled signatures
print(k.targetoptions)          # merged numba options
```

For a module-level kernel use `cm.kernels(Cls)` /
`kernel.signatures()` directly.
