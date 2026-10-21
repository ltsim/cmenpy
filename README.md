# cmenpy

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
![PyPI - Version](https://img.shields.io/pypi/v/cmenpy?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cmenpy?style=flat-square)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/cmenpy?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cmenpy?style=flat-square)
![Publish](https://img.shields.io/github/actions/workflow/status/ltsim/cmenpy/publish.yml?style=flat-square&logo=pypi&label=Publish)
![Testing](https://img.shields.io/github/actions/workflow/status/ltsim/cmenpy/test.yml?style=flat-square&logo=pytest&label=Testing)
![Typed](https://img.shields.io/github/actions/workflow/status/ltsim/cmenpy/type.yml?style=flat-square&logo=python&label=Typed)

**C**ompute **M**etaheuristic **En**gine for **P**ython (_cmenpy_ or _/siːmɛnpaɪ/_) is an extensible, flexible, and vectorized Python framework for designing and prototyping metaheuristic and population-based optimization algorithms.

It provides the clean abstractions necessary to build custom evolutionary algorithms, swarm intelligence methods, and hybrid heuristics from scratch — with absolute architectural control.

## Installation

```bash
pip install cmenpy
```

Requires Python `>=3.11`. Numba is **optional**: kernels compile when it is
installed (`pip install "cmenpy[numba]"`), and run as pure NumPy/Python when
it is not — or whenever a model asks with `numba=False`.

To develop or build the docs:

```bash
git clone https://github.com/ltsim/cmenpy.git
cd cmenpy
uv sync --extra dev        # tests, type checks, linters
uv sync --extra docs       # ... plus the MkDocs toolchain
```

## Quick example

```python
import numpy as np
import cmenpy as cm


@cm.declare(x=cm.Argument[int, (0, 3), 2])
def my_algorithm(args, epoch, ctx):
    pop = cm.PopulationSwarm(ctx.buff)
    pop.compute << ctx.rng.uniform(
        ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
    )
    # ... one epoch of your operator ...


best_fitness, best_x = my_algorithm(x=1).solve(
    lambda x: np.sum(x**2), cm.Bounds([(-1, 1)] * 3), epochs=150, pop_size=15
)
```

See the [Quickstart](https://ltsim.github.io/cmenpy/quickstart/) for both the function (`@cm.declare`) and class (`@cm.template`) styles.

## Core pillars

- **Vectorized data structures.** Population arrays, agent states, and search spaces use a flattened, vectorized layout with no per-individual objects, interfacing natively with NumPy and Numba.
- **Extensible architecture.** Custom operators, mutation rules, or entirely new paradigms require overriding only a few decoupled, well-defined methods.
- **Non-monolithic flexibility.** No visualization tools, file exporters, or rigid loggers — just the mathematical engine and its structural flow.
- **Research ready.** Licensed under the **GNU GPLv3**. Design, modify, and distribute your algorithms freely; derivative works stay open, keeping the software a public good for research and commercial development.

## Documentation

Full guides and the API reference are published at **[ltsim.github.io/cmenpy](https://ltsim.github.io/cmenpy/)**.

| Guide | Covers |
| ----- | ------ |
| [Quickstart](https://ltsim.github.io/cmenpy/quickstart/) | Solve your first problem in minutes |
| [Models](https://ltsim.github.io/cmenpy/models/) | `@cm.declare` vs `@cm.template`, the `ctx` object |
| [Build a custom model](https://ltsim.github.io/cmenpy/tutorial-custom-model/) | Step-by-step tutorial implementing your own optimizer |
| [Numba kernels](https://ltsim.github.io/cmenpy/compile/) | `@cm.precompile`, `@cm.vectorize`, `@cm.reduce` |
| [Population](https://ltsim.github.io/cmenpy/population/) | `PopulationSwarm`, buffers, selection, ranking |
| [Target functions](https://ltsim.github.io/cmenpy/target/) | Objectives, NFE budget, vectorized targets |
| [Examples](https://ltsim.github.io/cmenpy/examples/) | Runnable algorithms in `example/` |
| [API reference](https://ltsim.github.io/cmenpy/api/) | Full reference for every public symbol, by category |
| [Changelog](https://ltsim.github.io/cmenpy/changelog/) | Release notes |

To preview locally: `uv sync --extra docs && .venv/bin/mkdocs serve`.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the list of [contributors](CONTRIBUTORS.md).

---

*Developed by [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026. Licensed under [GNU GPLv3](LICENSE.md).*
