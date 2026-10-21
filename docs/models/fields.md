# Fields and markers

Model fields are declared as class annotations (or `@cm.declare` keyword
declarations) using three subscript-only markers. Each marker builds a
`typing.Annotated` type whose metadata carries the field's rules — that is
why fields are validated at declaration/binding time and read like plain
attributes afterwards.

```python
class MyAlgorithm:
    w: cm.Argument[float, (0.1, 1.0), 0.5]   # tunable
    pbest: cm.Variable[np.ndarray]           # internal state
    P: cm.Constant[int, 100]                 # fixed
```

## `cm.Argument` — tunable parameters

| Form | Meaning |
| ---- | ------- |
| `cm.Argument[type]` | required at binding; no range |
| `cm.Argument[type, default]` | default value when not bound |
| `cm.Argument[type, (min, max)]` | range-checked, required |
| `cm.Argument[type, (min, max), default]` | range-checked with a default |

```python
@cm.template
class MyAlgorithm:
    a: cm.Argument[int, (1, 100)]            # must be bound
    b: cm.Argument[int, (1, 100), 5]         # defaults to 5


MyAlgorithm(a=3)             # ok
MyAlgorithm()                # TypeError: Missing required argument: 'a'
MyAlgorithm(a=0)             # ValueError: Value 0 out of range for 'a'
MyAlgorithm(a="x")           # TypeError: Field 'a' must be of type int
```

An unsupported subscript raises `TypeError` listing the four valid forms.

Validation is **stricter for range-declared fields**: those get the type
check, the range check and the default fallback. A field declared as
`cm.Argument[type]` (or `cm.Variable[type]`) is stored as given — its
annotation documents intent but is not enforced at construction.

In the function style, binding validates **every** declared field's type, and
its range when one is declared; errors surface as `TypeError` / `ValueError`
(see [declare](declare.md#errors-at-binding)).

## `cm.Variable` — internal state

```python
w: cm.Variable[np.ndarray]
```

`Variable` marks state that is not a tunable parameter: no range, no
default, nothing required. It is stored as `None` until you assign it (in
`initialize` or `evolve`). Use it for personal-best arrays, weights, caches —
anything the algorithm maintains rather than the user configures.

## `cm.Constant` — fixed values

```python
P: cm.Constant[int, 100]
```

- The value is **type-checked once at declaration**: a mismatch raises
  `TypeError` (with numeric widening — an `int` value satisfies a `float`
  constant, mirroring the field casters).
- It is installed as a read-only class descriptor: readable as `self.P` and
  `MyAlgorithm.P`, but reassignment or deletion raises `AttributeError`.
- It is **never a constructor parameter** — passing it to the constructor
  raises `TypeError`.
- Inherited through the MRO (an Operator Module's constants reach every
  model that mixes it in), and a subclass may redeclare the name to override
  the value.

Constants are a **class-style feature**: `@cm.declare` rejects them at
declaration with `TypeError: Constant field 'P' is not supported by
cm.declare; ...`.

## Choosing a marker

| You want… | Use |
| --------- | --- |
| a knob the user sets (`lr`, `pop factor`) | `cm.Argument` |
| a knob with a sensible default | `cm.Argument[type, default]` or the range+default form |
| algorithm state the run maintains | `cm.Variable` |
| a fixed hyperparameter shared by every run/model | `cm.Constant` |

## Gotchas

- **Annotations are resolved at declaration** (`typing.get_type_hints` with
  `include_extras=True`), so stringified annotations and
  `from __future__ import annotations` work — the marker objects must be
  importable in the class's module namespace.
- **`initialize` and `evolve` are excluded** from field processing; a field
  named like them is not bound.
- **Positional binding** follows declaration order (constants skipped) —
  prefer keywords ([template](template.md#instantiation)).
- **Numeric widening** applies to constants (`int` → `float`); the
  range-declared field checks use strict `isinstance`.
