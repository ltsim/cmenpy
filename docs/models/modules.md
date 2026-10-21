# Operator modules

An **Operator Module** is a Ruby-style mixin: a bundle of kernels, helper
methods and fields that any number of `@cm.template` models can share by
inheriting from it.

```python
import numpy as np

import cmenpy as cm


@cm.module
class MutationOps:
    """A reusable Gaussian-mutation operator module."""

    scale: cm.Constant[float, 0.3]              # shared, read-only

    @cm.vectorize
    def clip(x: cm.Array, lo: cm.Scalar, hi: cm.Scalar):
        return min(max(x, lo), hi)

    def mutate(self, parents, rng):
        noise = rng.standard_normal(parents.shape)
        return self.clip(parents + self.scale * noise, -1.0, 1.0)
```

```python
@cm.template
class HillClimber(MutationOps):
    def evolve(self, e, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        trial = cm.PopulationSwarm(ctx.buff.snapshot)
        trial.compute << self.mutate(~pop.extract, ctx.rng)

        mask = cm.is_best((trial.F, pop.F), ctx.sense)
        pop.assign[mask] << trial.access[mask]
```

`HillClimber` now has `self.scale`, `self.clip` and `self.mutate` without
duplicating a line.

## What `@cm.module` does

- keeps the **class identity** — it returns the same class;
- **materializes `cm.Constant` annotations** as read-only descriptors
  immediately, so `MutationOps.scale` works even before any model includes
  the mixin;
- rejects non-classes with `TypeError` (a `@cm.template` model instance is
  not reusable as a mixin).

Mixin application is **ordinary inheritance**: a module can hold anything a
base class can — `cm.Argument` / `cm.Variable` / `cm.Constant` fields, all
three kernel decorators, plain helper methods.

## MRO rules

Collisions resolve through the normal MRO, **leftmost base wins**:

```python
@cm.template
class Hybrid(FastMutation, SlowMutation):   # FastMutation's operators win
    ...
```

A subclass redefinition always wins over any base, module or not. Constants
follow the same rule, and redeclaring the name in a subclass overrides the
inherited value.

## Kernels and compile options

Kernels declared in a module are inherited exactly like methods: called
through the model instance (`self.clip(...)`), they compile with the **model's**
compile options; accessed on the module class they use the decorator's.
`model.prepare()` / `cm.prepare` walk the MRO, so inherited kernels are
reported and warmed up with the model's options too.

```python
model = HillClimber(seed=1, fastmath=True)
model.prepare()          # {"clip": [...], ...} — includes the module's clip
```

See [Compilation options](../compile/options.md).

## Composition patterns

- **Keep modules stateless.** Let the model own the state (`cm.Variable`) and
  the module own behavior plus `cm.Constant` hyperparameters.
- **One concern per module** — `MutationOps`, `SelectionOps`, `CrossoverOps`
  — composed by listing bases: `class GA(SelectionOps, CrossoverOps):`.
- **Temper constants in the model** when the fixed value is only a baseline:

  ```python
  step = self.scale * (1.0 - e / ctx.epochs)     # Annealer in the example
  ```

## When to use a module vs a plain base class

`@cm.module` is not required for inheritance — a plain base class works
almost the same. Use it when you want constants installed (and validated) at
decoration time, so `MutationOps.scale` is reachable on the mixin itself
before any model includes it. Use a plain base class when the "mixin" is
really a template itself.

## Full example

`example/operator_module.py` defines `MutationOps` once and builds two
models (`HillClimber`, `Annealer`) on top of it, runnable side by side.
