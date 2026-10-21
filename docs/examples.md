# Examples

`example/` holds runnable algorithms and demos. It is excluded from the
built wheel. Every optimizer solves a reference test function from
[`opfunu-core`](https://github.com/ltsim/opfunu-core), and
`engineering_demo.py` solves a constrained problem from
[`enoppy-core`](https://github.com/ltsim/enoppy-core) — install both with:

```bash
uv sync --extra benchmarks
```

Optimizer implementations are grouped by algorithm family under
`example/optimizers/`; files that demonstrate the API itself stay at the top
level.

## Getting started

| File | Shows |
| ---- | ----- |
| `hello_world_v1.py` | `@cm.declare` function style |
| `hello_world_v2.py` | `@cm.template` class style + `@cm.vectorize` / `@cm.precompile` kernels |
| `operator_module.py` | `@cm.module` Operator Modules + `cm.Constant`, reused across two models |
| `reduce_demo.py` | `@cm.reduce` with annotation-inferred layout |
| `engineering_demo.py` | constrained engineering problem via `enoppy-core` |

## Optimizers

Each directory under `example/optimizers/` collects the implementations of one
well-known optimizer.

### `de/` — Differential Evolution

| File | Algorithm |
| ---- | --------- |
| `de/de_algorithm.py` | Differential Evolution (jDE-style self-adaptive) |
| `de/jade_algorithm.py` | JADE with external archive |
| `de/lshade_algorithm.py` | LSHADE with success-history adaptation + linear population-size reduction |

### `gwo/` — Grey Wolf Optimizer

| File | Algorithm |
| ---- | --------- |
| `gwo/gwo_algorithm.py` | Grey Wolf Optimizer (pure Python reference) |
| `gwo/gwo_algorithm_vectorized.py` | GWO with numba kernels (`GreyWolfOptimizer(parallel=True)`) |

### One-file families

| Directory | File | Algorithm |
| --------- | ---- | --------- |
| `pso/` | `pso/pso_algorithm.py` | Particle Swarm Optimization |
| `ga/` | `ga/ga_algorithm.py` | Real-coded Genetic Algorithm (self-adaptive, elitist) |
| `sa/` | `sa/sa_algorithm.py` | Simulated Annealing as a population of chains |
| `rs/` | `rs/random_search.py` | Adaptive Random Search (Luus–Jaakola style) |

Every one of them solves an [`opfunu-core`](https://github.com/ltsim/opfunu-core)
function through the `f.evaluate` / `f.bounds` pattern described in
[Test functions](benchmarks.md).

Run any of them from the repo root, e.g.:

```bash
.venv/bin/python example/reduce_demo.py
.venv/bin/python example/hello_world_v1.py
.venv/bin/python example/optimizers/de/jade_algorithm.py
```
