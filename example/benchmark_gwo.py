"""Benchmark: original function-based GWO vs the precompiled numba GWO.

Run from the repository root::

    .venv/bin/python -m example.benchmark_gwo

The numba kernels declare typed markers, so they compile eagerly when the class
is defined (before ``solve``). A cold run (no warmup) therefore matches the warm
runs, and the one-time compilation cost is paid at import time.
"""

import statistics
import subprocess
import sys
import time
import typing

from opfunu.name_based import Ackley01

import cmenpy as cm
from example.gwo_algorithm import gwo
from example.gwo_algorithm_vectorized import GreyWolfOptimizer

EPOCHS = 150
POPSIZE = 75
NDIM = 30
RUNS = 5

Model = typing.Callable[[int], typing.Any]


def evaluate(build: Model, f: Ackley01, seed: int) -> tuple[float, float]:
    model = build(seed)

    start = time.perf_counter()
    fitness, _ = model.solve(f.evaluate, f.bounds, EPOCHS, POPSIZE)
    elapsed = time.perf_counter() - start

    return elapsed, float(fitness)


def bench(name: str, build: Model, f: Ackley01, runs: int) -> None:
    cold_time, _ = evaluate(build, f, 0)

    times: list[float] = []
    fitnesses: list[float] = []

    for seed in range(1, runs + 1):
        elapsed, fitness = evaluate(build, f, seed)
        times.append(elapsed)
        fitnesses.append(fitness)

    print(
        f"{name:<10} "
        f"cold={cold_time:6.3f}s  "
        f"mean={statistics.mean(times):6.3f}s  "
        f"min={min(times):6.3f}s  "
        f"fitness={statistics.mean(fitnesses):.3e}"
    )


def precompile_cost() -> float:
    """Compile the numba GWO in a fresh process and return the elapsed time."""
    code = (
        "import time; t=time.perf_counter(); "
        "import example.gwo_algorithm_vectorized as m; "
        "import cmenpy as cm; cm.prepare(m.GreyWolfOptimizer); "
        "print(time.perf_counter()-t)"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code], cwd=".", text=True
    )

    return float(output.strip().splitlines()[-1])


if __name__ == "__main__":
    f = Ackley01(ndim=NDIM)

    print(f"Ackley ndim={NDIM}  epochs={EPOCHS}  pop={POPSIZE}  runs={RUNS}\n")

    kernels = cm.prepare(GreyWolfOptimizer)
    print("precompiled kernels:", ", ".join(sorted(kernels)))
    print(f"one-time precompile (fresh process): {precompile_cost():.3f}s\n")

    bench("GWO", lambda seed: gwo(seed=seed), f, RUNS)
    bench("GWO_numba", lambda seed: GreyWolfOptimizer(seed=seed), f, RUNS)
