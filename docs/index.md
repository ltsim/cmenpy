# cmenpy

**C**ompute **M**etaheuristic **En**gine for **P**ython (_cmenpy_ or _/siːmɛnpaɪ/_) is an
extensible, flexible and vectorized Python framework for designing and prototyping
metaheuristic and population-based optimization algorithms.

## Core pillars

- **Vectorized data structures:** population arrays, agent states and search spaces
  use a flattened, vectorized layout with no heavy per-individual objects, and
  interoperate natively with NumPy and Numba.
- **Extensible architecture:** custom operators, mutation rules or whole new paradigms
  only require overriding a few decoupled methods.
- **Non-monolithic flexibility:** no visualization tools, file exporters or rigid
  loggers — just the mathematical engine and its structural flow.
- **Research ready:** licensed under **GNU GPLv3**.

## Install

```bash
pip install cmenpy          # runtime: numpy + numba only
```

For development (tests, type checks, docs):

```bash
uv sync --extra dev         # runtime + dev tools
uv sync --extra docs        # ... plus the MkDocs toolchain
```

Requires Python `>=3.11`.

## Layout

| Path             | Contents |
| ---------------- | -------- |
| `cmenpy/`        | The package: `compile`, `model`, `population`, `target`, `kernel`, `context`, `functools`, `bounds` |
| `tests/`         | Test suite mirroring the package (`pytest`) |
| `example/`       | Runnable algorithms and demos (not shipped in the wheel) |
| `docs/`          | This site (MkDocs + Material) |

## Next steps

- [Quickstart](quickstart.md) — solve your first problem in minutes.
- [Numba kernels](compile.md) — `@cm.precompile`, `@cm.vectorize`, `@cm.reduce`.
- [Models](models.md) — `@cm.declare` vs `@cm.template`.
- [API reference](api.md) — full autodoc for every public symbol.

Developed by [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026.
