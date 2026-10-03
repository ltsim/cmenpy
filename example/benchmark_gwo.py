"""Benchmark: original function-based GWO vs the precompiled numba GWO.

Sweeps the ``ndim x pop_size`` grid on Ackley01 and reports, for every cell:

* wall time of both models (min and p50 over ``RUNS`` seeds),
* the per-epoch cost of the work *both* models share (fitness evaluation and the
  random draw), which is what caps the speedup at high dimensions,
* the same comparison with a vectorized objective (``cm.vectorized_target``),
  i.e. without the per-individual ``np.apply_along_axis`` dispatch,
* the numba thread count, because the ``prange`` in ``hunt`` costs more than it
  saves at these problem sizes.

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

import numba as nb
import numpy as np
from opfunu.name_based import Ackley01

import cmenpy as cm
from example.gwo_algorithm import gwo
from example.gwo_algorithm_vectorized import GreyWolfOptimizer

EPOCHS = 150
RUNS = 10
NDIMS = (30, 120, 300, 500)
POPSIZES = (25, 75, 150)

# `hunt` parallelizes over `pop` iterations of ~40us each, so the thread wake-up
# outweighs the work done. One thread keeps the timings stable and comparable.
THREADS = 1
THREAD_SWEEP = (1, 4, 12)
THREAD_SWEEP_DIMS = (120, 500)
THREAD_SWEEP_RUNS = 7

VECTORIZED_AB = True

Model = typing.Callable[[int], typing.Any]

MODELS: tuple[tuple[str, Model], ...] = (
    ("GWO", lambda seed: gwo(seed=seed)),
    ("GWO_numba", lambda seed: GreyWolfOptimizer(seed=seed)),
)

# `@cm.template` returns the model wrapper and keeps the kernels as static
# methods on the wrapped class, which is what holds the compiled callables.
KERNEL_CLASS = GreyWolfOptimizer()._TemplateOptimizerModel__cls_model
hunt = getattr(KERNEL_CLASS, "hunt")
scale = getattr(KERNEL_CLASS, "scale")
sweep_coeff = getattr(KERNEL_CLASS, "sweep")


def measure(
    build: Model,
    f: Ackley01,
    target: typing.Callable,
    pop: int,
    seed: int,
    epochs: int,
) -> tuple[float, float]:
    model = build(seed)

    start = time.perf_counter()
    fitness, _ = model.solve(target, f.bounds, epochs, pop)
    elapsed = time.perf_counter() - start

    return elapsed, float(fitness)


def best_of(fn: typing.Callable[[], typing.Any], repeats: int = 5) -> float:
    """Return the fastest of ``repeats`` calls, discarding scheduler noise."""
    fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    return min(times)


def numpy_hunt(X, leaders, A, C):
    """Position update of ``gwo_algorithm.py``: one Python iteration per wolf."""
    out = np.empty_like(X)

    for i, p in enumerate(X):
        D = [b - A[i][j] * np.abs(C[i][j] * b - p) for j, b in enumerate(leaders)]
        out[i] = np.sum(D, axis=0) / len(leaders)

    return out


def diagnose(f: Ackley01, ndim: int, pop: int) -> dict[str, float]:
    """Per-epoch cost of the shared work and of each model's position update."""
    rng = np.random.default_rng(0)
    low, up = f.bounds[:, 0], f.bounds[:, 1]

    X = rng.uniform(low, up, (pop, ndim))
    leaders = X[:3]
    A = rng.random((pop, 3, ndim))
    C = rng.random((pop, 3, ndim))
    rnd = rng.random((pop, 3, ndim))

    return {
        "eval": best_of(lambda: np.apply_along_axis(f.evaluate, 1, X)),
        "vec_eval": best_of(lambda: vectorized_ackley(X)),
        "rnd": best_of(lambda: rng.random((pop, 3, ndim))),
        "coef": best_of(lambda: scale(rnd, 1.5)) + best_of(lambda: sweep_coeff(rnd)),
        "np_hunt": best_of(lambda: numpy_hunt(X, leaders, A, C), repeats=3),
        "nb_hunt": best_of(lambda: hunt(X, leaders, A, C)),
    }


def vectorized_ackley(x: np.ndarray) -> np.ndarray:
    """``Ackley01`` over a whole ``(pop, ndim)`` block instead of one row at a time."""
    u = np.sum(x**2, axis=1)
    v = np.sum(np.cos(2.0 * np.pi * x), axis=1)

    return (
        -20.0 * np.exp(-0.2 * np.sqrt(u / x.shape[1]))
        - np.exp(v / x.shape[1])
        + 20.0
        + np.exp(1.0)
    )


def sweep(
    f: Ackley01,
    target: typing.Callable,
    pop: int,
    runs: int,
    epochs: int,
) -> tuple[dict, dict]:
    """Interleave both models seed by seed so machine drift hits them equally."""
    times: dict[str, list[float]] = {name: [] for name, _ in MODELS}
    fitnesses: dict[str, list[float]] = {name: [] for name, _ in MODELS}

    for _, build in MODELS:
        measure(build, f, target, pop, 0, epochs)  # cold run, kernels compiled at import

    for seed in range(1, runs + 1):
        for name, build in MODELS:
            elapsed, fitness = measure(build, f, target, pop, seed, epochs)

            times[name].append(elapsed)
            fitnesses[name].append(fitness)

    return times, fitnesses


def summarize(times: dict, fitnesses: dict) -> dict[str, dict[str, float]]:
    return {
        name: {
            "min": min(times[name]),
            "p50": statistics.median(times[name]),
            "fit": statistics.median(fitnesses[name]),
            "fit_min": min(fitnesses[name]),
            "fit_max": max(fitnesses[name]),
        }
        for name, _ in MODELS
    }


def sweep_grid(
    target_of: typing.Callable[[Ackley01], typing.Callable] = lambda f: f.evaluate,
    dims: tuple[int, ...] = NDIMS,
    pops: tuple[int, ...] = POPSIZES,
    runs: int = RUNS,
    epochs: int = EPOCHS,
) -> dict[tuple[int, int], dict[str, dict[str, float]]]:
    results: dict[tuple[int, int], dict[str, dict[str, float]]] = {}

    for ndim in dims:
        f = Ackley01(ndim=ndim)
        target = target_of(f)

        for pop in pops:
            times, fitnesses = sweep(f, target, pop, runs, epochs)
            results[(ndim, pop)] = summarize(times, fitnesses)

    return results


def report(title: str, results: dict) -> None:
    print(f"\n{title}")
    print(
        f"{'ndim':>5} {'pop':>5} | {'GWO min':>9} {'p50':>9} | "
        f"{'numba min':>10} {'p50':>9} | {'x min':>7} {'x p50':>7} |"
        f" {'fitness GWO':>22} {'fitness numba':>22}"
    )
    print("-" * 122)

    for (ndim, pop), row in results.items():
        gwo, numba = row["GWO"], row["GWO_numba"]

        print(
            f"{ndim:>5} {pop:>5} | {gwo['min']:9.4f} {gwo['p50']:9.4f} | "
            f"{numba['min']:10.4f} {numba['p50']:9.4f} | "
            f"{gwo['min'] / numba['min']:6.2f}x {gwo['p50'] / numba['p50']:6.2f}x | "
            f"{gwo['fit']:11.4e} [{gwo['fit_min']:.2e}] "
            f"{numba['fit']:11.4e} [{numba['fit_min']:.2e}]"
        )


def report_diagnostics(
    results: dict,
    dims: tuple[int, ...] = NDIMS,
    pops: tuple[int, ...] = POPSIZES,
    epochs: int = EPOCHS,
) -> None:
    print(f"\nCosto por epoca (ms) — el bloque 'compartido' es identico en ambos modelos")
    print(
        f"{'ndim':>5} {'pop':>5} | {'eval':>8} {'vec_eval':>9} {'rnd':>8} "
        f"{'np_hunt':>9} {'nb_hunt':>9} |"
        f" {'epoca GWO':>10} {'epoca numba':>12} | {'% eval GWO':>11} {'% eval numba':>13}"
    )
    print("-" * 130)

    for ndim in dims:
        f = Ackley01(ndim=ndim)

        for pop in pops:
            cost = diagnose(f, ndim, pop)
            row = results[(ndim, pop)]

            epoch_gwo = row["GWO"]["min"] / epochs * 1000
            epoch_numba = row["GWO_numba"]["min"] / epochs * 1000

            print(
                f"{ndim:>5} {pop:>5} | {cost['eval'] * 1000:8.3f} "
                f"{cost['vec_eval'] * 1000:9.3f} {cost['rnd'] * 1000:8.3f} "
                f"{cost['np_hunt'] * 1000:9.3f} {cost['nb_hunt'] * 1000:9.3f} |"
                f" {epoch_gwo:10.3f} {epoch_numba:12.3f} |"
                f" {cost['eval'] * 1000 / epoch_gwo * 100:10.1f}% "
                f"{cost['eval'] * 1000 / epoch_numba * 100:12.1f}%"
            )


def report_speedup_trend(results: dict) -> None:
    print("\nSpeedup (numba) por combinacion — min / p50")
    print(f"{'ndim':>5} | " + " ".join(f"{'pop=' + str(pop):>14}" for pop in POPSIZES))
    print("-" * (7 + 16 * len(POPSIZES)))

    for ndim in NDIMS:
        cells = []

        for pop in POPSIZES:
            gwo, numba = results[(ndim, pop)]["GWO"], results[(ndim, pop)]["GWO_numba"]
            cells.append(f"{gwo['min'] / numba['min']:6.2f}x /{gwo['p50'] / numba['p50']:6.2f}x")

        print(f"{ndim:>5} | " + " ".join(f"{cell:>14}" for cell in cells))


def thread_sweep() -> None:
    print(f"\nHilos de numba (solo GWO_numba; numpy no cambia)")

    header = f"{'ndim':>5} {'pop':>5} |"
    rows: dict[tuple[int, int], list[str]] = {}

    for ndim in THREAD_SWEEP_DIMS:
        f = Ackley01(ndim=ndim)

        for pop in POPSIZES:
            rows[(ndim, pop)] = []

            for threads in THREAD_SWEEP:
                nb.set_num_threads(threads)
                _, build = MODELS[1]

                measure(build, f, f.evaluate, pop, 0, EPOCHS)

                times = [
                    measure(build, f, f.evaluate, pop, seed, EPOCHS)[0]
                    for seed in range(1, THREAD_SWEEP_RUNS + 1)
                ]

                rows[(ndim, pop)].append(
                    f"{min(times):7.4f}/{statistics.median(times):7.4f}"
                )

    nb.set_num_threads(THREADS)

    header += "".join(f"{'hilos=' + str(t):>15}" for t in THREAD_SWEEP)
    print(header)
    print("-" * len(header))

    for (ndim, pop), cells in rows.items():
        print(f"{ndim:>5} {pop:>5} |" + "".join(f"{cell:>15}" for cell in cells))

    print("  (min/p50 en segundos)")


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
    nb.set_num_threads(THREADS)

    print(
        f"Ackley01  epochs={EPOCHS}  runs={RUNS}  dims={NDIMS}  pops={POPSIZES}  "
        f"numba_threads={THREADS}"
    )

    kernels = cm.prepare(GreyWolfOptimizer)
    print("precompiled kernels:", ", ".join(sorted(kernels)))
    print(f"one-time precompile (fresh process): {precompile_cost():.3f}s")

    results = sweep_grid()
    report("GWO vs GWO_numba", results)
    report_speedup_trend(results)
    report_diagnostics(results)

    if VECTORIZED_AB:
        vectorized = sweep_grid(lambda f: cm.vectorized_target(vectorized_ackley))

        report("GWO vs GWO_numba — objetivo vectorizado (cm.vectorized_target)", vectorized)

        print("\nGanancia del objetivo vectorizado (x)")
        print(
            f"{'ndim':>5} | "
            + " ".join(f"{'pop=' + str(pop):>14}" for pop in POPSIZES)
        )
        print("-" * (7 + 16 * len(POPSIZES)))

        for ndim in NDIMS:
            cells = []

            for pop in POPSIZES:
                base = results[(ndim, pop)]["GWO_numba"]["min"]
                fast = vectorized[(ndim, pop)]["GWO_numba"]["min"]
                cells.append(f"{base / fast:13.2f}x")

            print(f"{ndim:>5} | " + " ".join(f"{cell:>14}" for cell in cells))

    thread_sweep()

    print(
        "\nNota: GWO y GWO_numba no son bit-identicos con la misma semilla — el primero\n"
        "sortea A y C en dos llamadas a rng.random, el segundo sortea una sola matriz y\n"
        "deriva C = 2r de ella. El fitness compara distribuciones, no ganador; la\n"
        "comparacion valida es la de tiempo."
    )
