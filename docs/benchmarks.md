# Test functions

`cmenpy` is deliberately an **engine** — it ships no benchmark suite. For
reference test functions, use the LTSIM companions:

| Library | Import | Domain | Size |
| ------- | ------ | ------ | ---- |
| [`opfunu-core`](https://github.com/ltsim/opfunu-core) | `import opfunu` | numerical benchmarks (classical + CEC 2005–2022) | 500+ functions |
| [`enoppy-core`](https://github.com/ltsim/enoppy-core) | `import enoppy` | real-world engineering problems (constrained) | 50+ problems |

Both are GPL v3, maintained forks of the original `opfunu` / `enoppy`
libraries. Install them with `cmenpy`:

```bash
pip install "cmenpy[benchmarks]"
# or, in this repository:
uv sync --extra benchmarks
```

## Numerical benchmarks (opfunu-core)

Every opfunu function exposes `evaluate(x)` (one solution vector in, one
fitness out) and `bounds` — a sequence of `(low, up)` pairs that `cm.Bounds`
accepts directly. The two together plug into any model:

```python
from opfunu.name_based import Ackley01

import cmenpy as cm


f = Ackley01(ndim=5)                # 5-D Ackley: global minimum 0 at origin

model = MyAlgorithm(seed=1)
best_fitness, best_x = model.solve(
    f.evaluate, f.bounds, epochs=200, pop_size=30
)
```

- Variable-dimension functions (`Ackley01`, `Griewank`, `Alpine01`, …) take
  an `ndim` keyword; fixed-dimension ones (`Booth`, `Zirilli`, …) validate
  the solution length at call time.
- Discover functions with `opfunu.FUNC_DATABASE` (name-based),
  `opfunu.CEC_DATABASE`, or
  `opfunu.get_functions_by_classname("Griewank")`.
- CEC functions shift/rotate their landscapes, so their minimum is not at the
  origin — read the printed best value against the function's reported
  optimum, not against zero.

### Why scoring is row-by-row here

`f.evaluate` scores **one solution per call**, so an opfunu objective runs
through the row-by-row path
([scoring modes](target/scoring.md)). For the vectorized fast path, write the block
form of the function once — and combine it with the compiled kernel pattern:

```python
import numpy as np

import cmenpy as cm


def ackley_block(X):
    n = X.shape[1]
    s1 = np.sum(X**2, axis=1)
    s2 = np.sum(np.cos(2.0 * np.pi * X), axis=1)
    return -20.0 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20.0 + np.e


model.solve(cm.vectorized_target(ackley_block), f.bounds, epochs=200, pop_size=30)
```

## Engineering problems (enoppy-core)

enoppy problems wrap objective **and constraints** into a penalized
evaluation. Two integration details:

- `problem.evaluate(x)` returns a **one-element array**, so coerce to float;
- bounds arrive as per-dimension `lb` / `ub` arrays; zip them for `cm.Bounds`.

```python
import numpy as np
from enoppy.paper_based.pdo_2022 import ThreeBarTrussProblem

import cmenpy as cm


prob = ThreeBarTrussProblem()


def objective(x):
    return float(np.asarray(prob.evaluate(x)).item())


bounds = cm.Bounds(list(zip(prob.lb, prob.ub)))

model = MyAlgorithm(seed=1)
best_fitness, best_x = model.solve(objective, bounds, epochs=300, pop_size=40)
```

- `prob.get_objs(x)` / `prob.get_cons(x)` expose the raw objective and
  constraint values if you want your own penalty
  (`prob = ThreeBarTrussProblem(f_penalty=...)`).
- Native constraint handling is planned (roadmap: constraint handling in the
  backlog); for now `evaluate`'s penalty is the supported contract.

The runnable version of this snippet is
[`example/engineering_demo.py`](https://github.com/ltsim/cmenpy/blob/main/example/engineering_demo.py).

## Where the repo uses them

- Every algorithm under `example/optimizers/` solves an `opfunu` function
  (`Ackley01`, `Griewank`, `ChungReynolds`, …) through `f.evaluate` /
  `f.bounds`; `example/engineering_demo.py` solves an `enoppy` problem.
- `tests/test_benchmarks.py` keeps both integrations honest in CI.
- The [Examples](examples.md) page lists what each demo runs against.
