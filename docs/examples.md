# Examples

`example/` holds runnable algorithms and benchmarks. It is excluded from the
built wheel, and some files import `opfunu`, which is **not** a declared
dependency — install it first:

```bash
uv pip install opfunu
```

## Getting started

| File | Shows |
| ---- | ----- |
| `hello_world_v1.py` | `@cm.declare` function style |
| `hello_world_v2.py` | `@cm.template` class style + `@cm.vectorize` / `@cm.precompile` kernels |
| `operator_module.py` | `@cm.module` Operator Modules + `cm.Constant`, reused across two models |
| `reduce_demo.py` | `@cm.reduce` with annotation-inferred layout |

## Optimizers

| File | Algorithm |
| ---- | --------- |
| `gwo_algorithm.py` | Grey Wolf Optimizer (pure Python reference) |
| `gwo_algorithm_vectorized.py` | GWO with numba kernels (`GreyWolfOptimizer(parallel=True)`) |
| `pso_algorithm.py` | Particle Swarm Optimization |
| `de_algorithm.py` | Differential Evolution (jDE-style self-adaptive) |
| `ga_algorithm.py` | Real-coded Genetic Algorithm (self-adaptive, elitist) |
| `jade_algorithm.py` | JADE with external archive |
| `lshade_algorithm.py` | LSHADE with success-history adaptation + linear population-size reduction |
| `sa_algorithm.py` | Simulated Annealing as a population of chains |
| `random_search.py` | Adaptive Random Search (Luus–Jaakola style) |

Run any of them from the repo root, e.g.:

```bash
.venv/bin/python example/reduce_demo.py
.venv/bin/python example/hello_world_v1.py
```
