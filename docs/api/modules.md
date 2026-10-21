# Modules

The package layout of `cmenpy`, one entry per subpackage. Only the module
docstrings are repeated here — the full autodoc for every symbol lives in
[Classes](classes.md) and [Functions](functions.md), and each entry links to
the sections that document it.

## cmenpy

The top-level package. Re-exports the whole public surface (see
[`__all__` in the source][repo-init]) and carries `__version__`.

[repo-init]: https://github.com/ltsim/cmenpy/blob/main/cmenpy/__init__.py

::: cmenpy
    options:
      members: false
      heading_level: 3

## cmenpy.compile

Numba kernel decorators, their typing markers and introspection helpers.
Re-exported at the top level, so `cmenpy.precompile` and
`cmenpy.compile.precompile` are the same object.

Documented in [Functions → Kernel decorators](functions.md#kernel-decorators),
[Functions → Signature markers](functions.md#signature-markers) and
[Functions → Introspection](functions.md#introspection).

::: cmenpy.compile
    options:
      members: false
      heading_level: 3

## cmenpy.model

Model declaration: the function-style `declare` and class-style `template`
decorators, the field markers, the optimizer classes and the builders that
turn a declaration into a runnable model.

Documented in [Classes → Optimizers](classes.md#optimizers),
[Classes → Field markers](classes.md#field-markers),
[Classes → Protocols](classes.md#protocols) and
[Functions → Declaration decorators](functions.md#declaration-decorators).

::: cmenpy.model
    options:
      members: false
      heading_level: 3

## cmenpy.population

Population abstractions: the `PopulationSwarm` handle, the stream operators
and pipelines, the matrix views, the roster/iteration bases and the agent
types.

Documented in [Classes → Population](classes.md#population) and
[Classes → Agents](classes.md#agents).

::: cmenpy.population
    options:
      members: false
      heading_level: 3

## cmenpy.kernel

The flat population buffer (`KernelBuffer`) plus the mask, size and ranking
support types.

Documented in [Classes → Buffer](classes.md#buffer) and
[Functions → Builders and factories](functions.md#builders-and-factories).

::: cmenpy.kernel
    options:
      members: false
      heading_level: 3

## cmenpy.target

Objective wrappers: the vectorized-target marker and the NFE-counting
`TargetFunction` that enforces the evaluation budget.

Documented in [Classes → Objectives](classes.md#objectives) and
[Functions → Declaration decorators](functions.md#declaration-decorators).

::: cmenpy.target
    options:
      members: false
      heading_level: 3

## cmenpy.bounds

The search-space description handed to `solve`.

Documented in [Classes → Search space](classes.md#search-space) and
[Functions → Builders and factories](functions.md#builders-and-factories).

::: cmenpy.bounds
    options:
      members: false
      heading_level: 3

## cmenpy.context

The per-epoch run context handed to models, and its owner.

Documented in [Classes → Runtime](classes.md#runtime).

::: cmenpy.context
    options:
      members: false
      heading_level: 3

## cmenpy.epoch

The epoch iterator that drives a run.

Documented in [Classes → Runtime](classes.md#runtime).

::: cmenpy.epoch
    options:
      members: false
      heading_level: 3

## cmenpy.tracker

The per-epoch history recorder and its snapshots (enabled with
`debug=True`).

Documented in [Classes → Runtime](classes.md#runtime).

::: cmenpy.tracker
    options:
      members: false
      heading_level: 3

## cmenpy.generator

The seeded random generator handed to models as `ctx.rng`.

Documented in [Classes → Runtime](classes.md#runtime).

::: cmenpy.generator
    options:
      members: false
      heading_level: 3

## cmenpy.functools

Sense-aware fitness comparators used by selection operators.

Documented in [Functions → Comparators](functions.md#comparators).

::: cmenpy.functools
    options:
      members: false
      heading_level: 3

## cmenpy.mixin

Reusable Operator Modules (`@cm.module`) for composing model behavior.

Documented in [Functions → Declaration decorators](functions.md#declaration-decorators).

::: cmenpy.mixin
    options:
      members: false
      heading_level: 3

## cmenpy.hints

Typing aliases, cast-spec markers and the structural protocols shared across
`cmenpy`. A pure numpy/typing leaf — it imports nothing from the rest of the
package.

Documented in [Classes → Protocols](classes.md#protocols) and
[Classes → Support types and aliases](classes.md#support-types-and-aliases).

::: cmenpy.hints
    options:
      members: false
      heading_level: 3
