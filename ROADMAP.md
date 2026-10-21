# Roadmap

This document tracks the direction of **cmenpy**: what is in the next release,
what is planned, and what is deliberately *not* going to be built here.

Status legend:

| Status | Meaning |
| ------ | ------- |
| `done` | shipped in a release |
| `in progress` | under development on the current milestone |
| `planned` | agreed scope, not started |
| `backlog` | desirable, not scheduled |

## Design principles

- **Vector first.** The hot paths stay flat NumPy/Numba arrays; object
  wrappers are an interface, never the storage.
- **Non-monolithic.** No plotting, no forced logging, no heavyweight
  dependencies. Persistence and reporting are **opt-in** adapters
  (`history=...`, `terminator=...`) that never run unless asked for.
- **Two declaration styles, one contract.** `@cm.declare` and `@cm.template`
  keep producing the same `solve(...)` surface; new features extend both or
  neither, never only one.
- **Backward compatibility inside a channel.** Additive changes ship freely;
  renames keep a deprecated alias for one release (see `cm.jit` →
  `cm.precompile`).

## 2026b2 — the next release

One active milestone that bundles the work of feature request
[#23](https://github.com/ltsim/cmenpy/issues/23) (split into
[#33](https://github.com/ltsim/cmenpy/issues/33) and
[#34](https://github.com/ltsim/cmenpy/issues/34)) with the run-control and
analysis features requested alongside it.

| Issue | Deliverable | Status |
| ----- | ----------- | ------ |
| [#33](https://github.com/ltsim/cmenpy/issues/33) | Composable terminators (`cmenpy.terminator`) | planned |
| [#34](https://github.com/ltsim/cmenpy/issues/34) | Decoders and `Space` (`cmenpy.decoder`, `cmenpy.space`) | planned |
| [#35](https://github.com/ltsim/cmenpy/issues/35) | Run history, RAM + disk (`cmenpy.history`) | planned |
| [#36](https://github.com/ltsim/cmenpy/issues/36) | Multiobjective Pareto core | planned |
| [#37](https://github.com/ltsim/cmenpy/issues/37) | Per-dimension bounds (`Bounds.lows/ups/sample/clip`) | planned |
| [#38](https://github.com/ltsim/cmenpy/issues/38) | Changelog restructure + this roadmap | in progress |
| [#39](https://github.com/ltsim/cmenpy/issues/39) | Test functions: examples, docs and tests use `opfunu-core` / `enoppy-core` | in progress |

### Terminators ([#33](https://github.com/ltsim/cmenpy/issues/33))

Stopping criteria become composable objects instead of hardcoded loop exits:

```python
terminator = (
    cm.MaxEvaluation(nfe=50_000) & cm.MaxEpoch(limit=10_000)
) | cm.MaxPatience(ms=60_000)

model.solve(f, bounds, epochs=10_000, pop_size=40, terminator=terminator)
```

Python cannot overload `&&` / `||`, so `&` and `|` are the supported spelling.
`MaxPatience(ms=...)` stops after N milliseconds without improvement and
`MaxTime(seconds=...)` is the hard wall-clock cap. `cm.Terminator(nfe=...)` is
kept as an alias of `MaxEvaluation`. Termination reasons surface in
`model.report`.

### Decoders and `Space` ([#34](https://github.com/ltsim/cmenpy/issues/34))

The decision space stays a flat continuous vector, but each slice of it can
now be interpreted as a different domain type — integers, strings,
categories, permutations, binaries — through `BaseDecoder` subclasses.

```python
space = cm.Space[
    cm.floats(low=0.0, up=1.0),
    cm.integers(low=0, up=10),
    cm.strings("A", "B", "C"),
]

# sugar form (a call, since Space[float=...] is not valid Python syntax)
space = cm.Space(float=(0.0, 1.0), string=("a", "b", "c"))
```

`solve(space=...)` is additive; `bounds=` keeps working, and `ctx.space`
exposes the decoder pipeline that runs immediately before fitness evaluation.

### Run history ([#35](https://github.com/ltsim/cmenpy/issues/35))

Per-epoch records become storable in RAM or on disk, opt-in per run:

```python
model.solve(..., history=True)             # in-memory summary
model.solve(..., history="run.jsonl")      # streamed summary
model.solve(..., history=cm.history.NPZ("run.npz", snapshots=True))
```

`debug=True` keeps its current full in-RAM behavior. A versioned reader
(`cm.history.load`) returns the records without imposing pandas.

### Multiobjective core ([#36](https://github.com/ltsim/cmenpy/issues/36))

Vector-valued objectives end-to-end: per-objective `sense`, fitness stored as
`(pop, m)` in the buffer, dominance/non-dominated sort/crowding in
`cmenpy.multiobjective`, Pareto fronts on the population
(`pop.front`, `pop.crowding`) and an NSGA-II reference example. Scalar runs
keep their exact shapes and performance.

### Per-dimension bounds ([#37](https://github.com/ltsim/cmenpy/issues/37))

`Bounds.low` / `Bounds.up` expose only the first dimension, so the common
`rng.uniform(bounds.low, bounds.up, ...)` idiom silently ignores unequal
intervals. `Bounds.lows`, `Bounds.ups`, `Bounds.sample()` and `Bounds.clip()`
fix this without breaking the existing accessors; examples and docs migrate to
the new helpers.

### Test functions ([#39](https://github.com/ltsim/cmenpy/issues/39))

The engine ships without a benchmark suite — reference test functions come
from the LTSIM companions [`opfunu-core`](https://github.com/ltsim/opfunu-core)
(500+ numerical functions: classical name-based plus every CEC competition)
and [`enoppy-core`](https://github.com/ltsim/enoppy-core) (50+ constrained
engineering problems). Every example solves an `opfunu` function,
`example/engineering_demo.py` shows `enoppy`, and
[tests/test_benchmarks.py](https://github.com/ltsim/cmenpy/blob/main/tests/test_benchmarks.py)
keeps both integrations honest. See the
[Test functions](https://ltsim.github.io/cmenpy/benchmarks/) guide.

## Backlog (not scheduled)

- Checkpoint / resume: serialize buffer, mask, RNG state and epoch so a run
  can continue from disk.
- MOO quality indicators: hypervolume, IGD, front spread.
- Constraint handling (penalty / feasibility rules) on top of decoders.
- Batched and parallel target evaluation.
- A reproducible benchmark suite (CEC-style functions).
- Polished `UserInterrupt` signal handling.

## Non-goals

cmenpy will stay a **design and prototyping engine**. We will not add:
plotting or visualization, rigid logging frameworks, a model zoo of named
algorithms as a core dependency, or non-GPU/non-NumPy numeric backends.
Those belong in companion packages.

## Proposing work

Open an issue first — see
[CONTRIBUTING.md](https://github.com/ltsim/cmenpy/blob/main/CONTRIBUTING.md). Ideas that are
adopted land in the 2026b2 table above or in the backlog, and the changelog's
`Unreleased` section is where they become visible.
