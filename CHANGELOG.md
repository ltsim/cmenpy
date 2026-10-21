# Changelog

All notable changes to **cmenpy** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses calendar versioning (`YYYYxN`): the year, a release
channel (`a` = alpha, `b` = beta), and an increment within that channel.

The `Unreleased` section collects changes targeting the next release. See the
[Roadmap](https://github.com/ltsim/cmenpy/blob/main/ROADMAP.md) for the scope
and ordering of that work.

## [Unreleased]

Targeted at **2026b2** (tracked in the
[roadmap](https://github.com/ltsim/cmenpy/blob/main/ROADMAP.md)).

### Added
### Added
- Numba is now an **optional backend**. The core install only requires NumPy;
- The `numba` compile option controls **whether compilation takes place**:
- Uncompiled kernels keep their specs (arguments are still auto-cast) and run
- `cm.prepare(model)` reports uncompiled kernels as `["<python>"]`.
- Models accept a `prepare` constructor option: `MyOptimizer(numba=True,
- Kernels cast array-protocol arguments **automatically, even when declared
- `from cmenpy import prange` binds numba's `prange` when the backend is


- Composable termination criteria in `cmenpy.terminator`: `MaxEvaluation`
  (with `cm.Terminator(nfe=...)` as an alias), `MaxEpoch`, `MaxPatience`,
  `MaxTime`, `EpochStagnation`, `TargetFitness`, `SpreadFitness` and
  `UserInterrupt`, combined with `&` (AND) and `|` (OR) — issue [#33](https://github.com/ltsim/cmenpy/issues/33).
- Opt-in run history in `cmenpy.history`: in-memory, JSONL, CSV and NPZ sinks
  with summary and full-snapshot levels, retention options, and a
  schema-versioned reader — issue [#35](https://github.com/ltsim/cmenpy/issues/35).
- Domain decoders and heterogeneous search spaces in `cmenpy.decoder` and
  `cmenpy.space` (floats, integers, strings, categories, sequences, binaries,
  booleans) with both the `cm.Space[...]` standard form and the
  `cm.Space(float=..., string=...)` sugar form — issue [#34](https://github.com/ltsim/cmenpy/issues/34).
- Multiobjective optimization: vector fitness columns, per-objective `sense`,
  Pareto dominance utilities, fronts and crowding distance, and an NSGA-II
  example — issue [#36](https://github.com/ltsim/cmenpy/issues/36).
- Per-dimension search-space helpers: `Bounds.lows`, `Bounds.ups`,
  `Bounds.sample()` and `Bounds.clip()` — issue [#37](https://github.com/ltsim/cmenpy/issues/37).
- Benchmark companions: reference test functions from
  [`opfunu-core`](https://github.com/ltsim/opfunu-core) (500+ numerical
  functions) and engineering problems from
  [`enoppy-core`](https://github.com/ltsim/enoppy-core) are now used across
  the examples, the test suite and the new
  [Test functions](https://ltsim.github.io/cmenpy/benchmarks/) guide page;
  install them with the new `benchmarks` extra — issue [#39](https://github.com/ltsim/cmenpy/issues/39).
- `example/engineering_demo.py`: a constrained engineering problem solved
  with `enoppy-core`.
- Roadmap documentation page.

### Changed

- The changelog has a single source of truth (`CHANGELOG.md`); the docs site
  includes it via a snippet instead of keeping a second copy that can drift.
- Every optimizer example solves an `opfunu-core` benchmark function instead
  of an inline hand-rolled objective.

### Fixed

- The NFE counter now charges **one evaluation per individual scored** for
  every way `pop.compute` selects rows. A NumPy row index — what `pop.best`,
  `pop.worst` and `pop.sort` return — used to be treated as a block: it spent
  `ndim` evaluations and then called the objective with a single solution,
  which raised for vectorized targets. `pop.insert`, `MutableAgent.solution`
  writes and the documented `pop.compute[pop.best] << solution` recipe are
  affected by the same dispatch and are fixed by it.
- Row-wise scoring no longer raises when a mask selects nothing (an emptied
  roster, or no row passing `cm.is_best`); such a selection now spends zero
  evaluations instead of failing inside `np.apply_along_axis`.
- `pop.compute` also accepts index iterators (`pop.compute[~pop]`) and the
  buffer's `KernelMask` (`pop.compute[ctx.buff.mask]`) as selectors.
- `TargetFunction.evaluate_matrix` rejects a bare solution vector with a clear
  message instead of spending `ndim` evaluations on it; a single solution is
  scored with `TargetFunction.evaluate`, which now dispatches a vectorized
  target as a one-row block.

## [2026b1] - unreleased

### Added
- `cm.cast(value, dtype=None, layout="A")` is the explicit way to turn a


- `@cm.module` turns a class into a reusable **Operator Module** (a Ruby-style
  mixin). It marks the class, materializes its `cm.Constant` annotations and
  keeps it usable as a plain base: `@cm.template class GA(Selection)` inherits
  kernels, helper methods and fields from `Selection`. Name collisions resolve
  through the MRO, and `cm.prepare(model)` now reports kernels inherited from
  an Operator Module (compilation walks `cls.__mro__` instead of `vars(cls)`).
- `cm.Constant[type, value]` declares a **fixed, read-only** field for
  `@cm.template` models and `@cm.module` Operator Modules. The value is checked
  at declaration, exposed through the class as a read-only descriptor
  (`self.name` / `Cls.name`), and is *never* constructor-bound — passing it to
  the model call raises `TypeError`. Constants flow into every implementing
  model via normal inheritance, so they work identically on a template and on
  an Operator Module.

### Documentation

- Guides reorganized into topic sections — Models, Population, Target
  functions and Numba kernels — each with per-category pages; all published
  URLs are preserved.
- API reference restructured into Classes, Functions and Modules.

----------------------------------------------------------------------

## [2026b0] - 2026-10-14

A decent and civilized version, compatible with GNU GPLv3.

### Added

- Numba decorators accept parameter annotations (`"Array[float, 2]"`,
  `"Scalar[float]"`, and mypy-clean `Annotated[..., cm.Array[...]]`), in
  addition to the positional `cm.Array` / `cm.Scalar` markers.
- `cm.precompile` (and the `cm.jit` alias) expose the `numba.njit` options
  `fastmath` and `parallel`; other numba options are forwarded as keywords.
- `cm.vectorize` and `cm.reduce` also accept `fastmath=True` and `parallel=True`
  (`parallel` maps to numba's `target="parallel"`).
- Decorators return a deferred `_Kernel`; kernels compile on first use (or via
  `cm.prepare`). A `@cm.template` model configures `parallel`, `fastmath` and
  `cache` at instantiation and they propagate to its kernels.
- `cm.best_index` / `cm.worst_index` sense-aware selectors in `cmenpy.functools`.
- Vectorized targets score a whole `(pop, ndim)` block in one call
  (`cmenpy.target.vectorized_target`).

### Changed

- `cm.jit` is renamed to `cm.precompile` (both use `numba.njit`); `cm.jit`
  remains as a deprecated alias that emits a `DeprecationWarning`.
- `cm.guvectorize` is renamed to `cm.reduce` (generalized ufunc). The layout
  is inferred from typed annotations (`Array[float, 'm,n']`, last parameter is
  the output); `cm.Contract[[inputs...], output]` remains as the explicit
  signature marker.

### Removed

- `cm.parallel`; use `cm.precompile(parallel=True)` standalone, or a
  model instantiated with `parallel=True` for kernels marked `parallel=True`.

### Documentation

- MkDocs site with guides and a full API reference, plus the
  "Build a custom model" tutorial.

----------------------------------------------------------------------

## [2026a0] - 2026-08-05

### Added

- First public version ("2026"): decorator-driven model declaration,
  vectorized population buffers and the basic engine structure, licensed
  under GNU GPLv3.

----------------------------------------------------------------------

## [0.0.0]

### Added

- Experimental test version.
- Modeled by decorators.
- Implementation of basic and primitive structure.
- Population vectorization to leverage SIMD.

[Unreleased]: https://github.com/ltsim/cmenpy/compare/2026b0...HEAD
[2026b1]: https://github.com/ltsim/cmenpy/compare/2026b0...HEAD
[2026b0]: https://github.com/ltsim/cmenpy/compare/2026a0...2026b0
[2026a0]: https://github.com/ltsim/cmenpy/commits/2026a0/
