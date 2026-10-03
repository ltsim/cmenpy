# GWO vs GWO-numba: dimension and population sweep report

Date: 2026-10-02
Branch: `feature/vectorize-numba`
Harness: `example/benchmark_gwo.py`

## TL;DR

- The numba GWO (`example/gwo_algorithm_vectorized.py`) is **1.22x to 2.56x faster**
  than the pure-Python GWO (`example/gwo_algorithm.py`) over the whole grid, and
  never slower.
- The single `hunt` kernel is **10.5x to 89x faster** than the Python loop it
  replaces, but it is only a small slice of an epoch, so end-to-end gains are
  much smaller.
- The speedup **collapses as dimension grows** (2.56x at `ndim=30, pop=150` down
  to 1.22x at `ndim=500, pop=150`) because the epoch is dominated by work that
  both variants share: objective evaluation and the `(pop, 3, ndim)` coefficient
  arrays.
- The new `cm.vectorized_target` feature attacks that shared cost directly and is
  the **larger, more general win**: up to **9.32x total** against the pure-Python
  GWO and up to **5.37x** against the same numba GWO with row-by-row evaluation.
- `parallel=True` on `hunt` **hurts** at these sizes: 12 threads are 20-41%
  slower than 1 thread. Keep `NUMBA_NUM_THREADS=1`.

## Environment

| | |
|---|---|
| CPU | Intel Core i7-8700 @ 3.20GHz, 6 cores / 12 threads |
| Cache | L1d 32 KiB/core, L2 256 KiB/core, L3 12 MiB shared |
| Python | 3.13.14 |
| NumPy | 2.5.2 |
| Numba | 0.67.0 |
| opfunu | 1.0.6 |

## Reproduce

```bash
.venv/bin/python -m example.benchmark_gwo
```

Runs in about 4m15s. All knobs are module constants at the top of
`example/benchmark_gwo.py`.

## Methodology

- Problem: `opfunu.name_based.Ackley01` on `[-35, 35]^ndim`.
- Budget: `EPOCHS=150`, `POPSIZE in {25, 75, 150}`, `NDIMS = {30, 120, 300, 500}`,
  `RUNS=10` seeds per cell.
- Both variants are **interleaved seed by seed** so that machine drift (thermal,
  scheduler) hits both equally.
- Reported statistic is `min` and `p50`, not the mean. The mean is contaminated by
  heavy-tail scheduler spikes; unfixed `NUMBA_NUM_THREADS` produced samples up to
  2.5x the baseline (0.0506s -> 0.1268s) in the same process.
- `NUMBA_NUM_THREADS=1` for the main grid (see the thread section for why). The
  pure-Python variant is single-threaded NumPy, so 1 thread is the fair comparison.
- **Caveat on fitness columns**: the two variants are not bit-identical under the
  same seed. The pure-Python version draws `A` and `C` in two `rng.random` calls
  while the numba version draws one matrix and derives `C = 2r` from it, so the RNG
  streams diverge. Fitness is comparable as a distribution, not as a winner; the
  valid comparison is time.

## Results

### End-to-end speedup (pure-Python / numba), min

| ndim | pop=25 | pop=75 | pop=150 |
|---:|---:|---:|---:|
| 30 | 2.14x | 2.47x | **2.56x** |
| 120 | 1.94x | 2.28x | 2.39x |
| 300 | 1.89x | 1.46x | 1.46x |
| 500 | 1.79x | 1.23x | 1.22x |

Two independent trends:

- **Population helps numba** while dimensions are low: at `ndim=30` the speedup
  grows 2.14x -> 2.56x as the population goes 25 -> 150, because the Python
  position loop is O(pop) in interpreter iterations.
- **Dimension is the ceiling**: the same population increase *reverses* the trend
  at `ndim=500` (1.79x -> 1.22x), because raising the population also scales the
  shared, non-vectorized work.

### Per-epoch cost breakdown (ms)

`eval` = `np.apply_along_axis(f.evaluate, 1, block)` (row by row, the default path);
`vec_eval` = the same Ackley over the whole block in one call;
`rnd` = `rng.random((pop, 3, ndim))`; `np_hunt` / `nb_hunt` = the position update of
each variant. `eval` and `rnd` are identical work in both variants.

| ndim | pop | eval | vec_eval | rnd | np_hunt | nb_hunt | epoch GWO | epoch numba | eval % GWO | eval % numba |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 25 | 0.244 | 0.028 | 0.007 | 0.447 | 0.009 | 0.871 | 0.408 | 28.0% | 59.9% |
| 30 | 75 | 0.754 | 0.055 | 0.019 | 1.342 | 0.017 | 2.345 | 0.950 | 32.2% | 79.3% |
| 30 | 150 | 1.448 | 0.097 | 0.036 | 2.670 | 0.030 | 4.539 | 1.772 | 31.9% | **81.7%** |
| 120 | 25 | 0.286 | 0.064 | 0.025 | 0.466 | 0.019 | 0.985 | 0.507 | 29.0% | 56.4% |
| 120 | 75 | 0.845 | 0.169 | 0.071 | 1.407 | 0.044 | 2.664 | 1.166 | 31.7% | 72.4% |
| 120 | 150 | 1.687 | 0.323 | 0.141 | 2.910 | 0.083 | 5.178 | 2.168 | 32.6% | 77.8% |
| 300 | 25 | 0.375 | 0.142 | 0.059 | 0.537 | 0.037 | 1.242 | 0.656 | 30.2% | 57.2% |
| 300 | 75 | 1.140 | 0.396 | 0.175 | 1.689 | 0.095 | 3.398 | 2.332 | 33.6% | 48.9% |
| 300 | 150 | 2.292 | 0.779 | 0.348 | 3.233 | 0.190 | 6.609 | 4.532 | 34.7% | 50.6% |
| 500 | 25 | 0.474 | 0.231 | 0.100 | 0.589 | 0.056 | 1.495 | 0.834 | 31.7% | 56.9% |
| 500 | 75 | 1.405 | 0.661 | 0.297 | 1.771 | 0.156 | 4.069 | 3.312 | 34.5% | 42.4% |
| 500 | 150 | 2.795 | 1.306 | 0.592 | 3.544 | 0.306 | 8.079 | 6.649 | 34.6% | 42.0% |

Reading the table:

- `nb_hunt` beats `np_hunt` by **10.5x** at (500, 25) and **89x** at (30, 150).
  The kernel is not the problem.
- Row-by-row evaluation is a **constant ~32% of the pure-Python epoch** and rises
  to **82% of the numba epoch** at (30, 150): once the kernel is fast, the
  evaluator is essentially the whole program.
- `eval` costs 9.6 µs per individual at (30, 150) and 18.6 µs at (500, 150). The
  per-individual Python dispatch is a fixed cost that only high dimensions amortize.
- `vec_eval` is 14.9x cheaper than `eval` at (30, 150) (1.448 -> 0.097 ms) but only
  2.1x cheaper at (500, 150) (2.795 -> 1.306 ms), where it becomes bandwidth-bound.

### Why the speedup plateaus at `ndim >= 300`

Every epoch materialises two `(pop, 3, ndim)` coefficient arrays, which are then
streamed twice (once by `scale`, once by `sweep`):

| pop | ndim | A + C | buffer |
|---:|---:|---:|---:|
| 25 | 120 | 141 KiB | 47 KiB |
| 75 | 120 | 422 KiB | 142 KiB |
| 150 | 120 | 844 KiB | 284 KiB |
| 75 | 500 | 1.72 MiB | 587 KiB |
| 150 | 500 | 3.43 MiB | 1.15 MiB |

The L2 is 256 KiB per core, so from `ndim=120, pop=75` onwards the coefficient
arrays no longer fit and the epoch becomes memory-bandwidth-bound. That is the
point where both the numba speedup and the vectorized-objective gain flatten: the
remaining cost is moving `pop x 3 x ndim` doubles, which neither change removes.

For reference, `rnd` alone is 0.592 ms/epoch at (500, 150) — almost **2x the cost
of the entire `hunt` kernel** (0.306 ms) and a bigger target than the kernel at
high dimensions.

### Vectorized objective A/B

Identical grid, with the objective declared through the new
`cm.vectorized_target` (see below), which replaces 150 per-row calls with one
matrix call per epoch.

Speedup (pure-Python / numba), min:

| ndim | pop=25 | pop=75 | pop=150 |
|---:|---:|---:|---:|
| 30 | 3.71x | 7.02x | **9.32x** |
| 120 | 2.97x | 4.45x | 5.10x |
| 300 | 2.47x | 2.93x | 1.55x |
| 500 | 2.16x | 1.21x | 1.19x |

Gain contributed by vectorizing the objective, numba variant only:

| ndim | pop=25 | pop=75 | pop=150 |
|---:|---:|---:|---:|
| 30 | 2.47x | 4.14x | 5.37x |
| 120 | 2.08x | 2.74x | 3.03x |
| 300 | 1.66x | 2.64x | 1.41x |
| 500 | 1.51x | 1.23x | 1.21x |

The fitness columns in the vectorized run are byte-for-byte the same as in the
default run, and the optimized trajectories are identical: the vectorized path is
a dispatch change, not an algorithmic one. This was verified independently —
`vectorized_ackley(X)` equals `np.apply_along_axis(Ackley01.evaluate, 1, X)` with
`max|diff| = 0.0` at `ndim` 2, 30 and 500.

### Numba threads

`hunt` parallelises over 25-150 iterations of roughly 40 µs each, so waking the
pool costs more than the work. Timings are `min/p50` seconds.

| ndim | pop | 1 thread | 4 threads | 12 threads |
|---:|---:|---:|---:|---:|
| 120 | 25 | **0.0745** | 0.0778 | 0.1031 |
| 120 | 75 | **0.1736** | 0.1800 | 0.2415 |
| 120 | 150 | **0.3266** | 0.3420 | 0.4600 |
| 500 | 25 | **0.1255** | 0.1258 | 0.1590 |
| 500 | 75 | **0.3232** | 0.3233 | 0.4215 |
| 500 | 150 | 0.9955 | **0.9872** | 1.1975 |

12 threads are 20-41% slower than 1 (the penalty is worst at `ndim=120`, where the
kernel is cheapest relative to the parallel-region overhead). The only cell where
4 threads edge ahead is (500, 150), by 0.8%, which is noise. The `prange` is not
paying for itself at any size tested, and leaving the thread count unset also
introduces the timing spikes described in the methodology.

## Is there an improvement?

Yes, on two counts, with an important qualification.

1. **The numba GWO is a strict improvement**: 1.22x-2.56x end-to-end, never a
   regression, with 1 thread. It is the better default implementation.

2. **The vectorized objective is the bigger and more general improvement**:
   up to 9.32x end-to-end and up to 5.37x over the already-vectorized numba
   variant, with bit-identical results. Unlike the kernel work, it removes cost
   from *both* variants and stays effective wherever evaluation is row-by-row.

3. **Qualification**: neither change helps much in the corner that matters most
   for scaling — high dimension *and* large population. At `ndim=500, pop=150` the
   numba GWO is 1.22x and the vectorized objective roughly cancels out (1.19x),
   because the epoch has become a memory-bandwidth problem over the coefficient
   arrays, and `rnd` alone is nearly twice the cost of `hunt`.

## Recommended next steps

- **Vectorize the coefficient generation.** Fold `rng.random`, `scale` and `sweep`
  into a single kernel that writes `A` and `C` in place, or better, into registers
  inside `hunt`, removing two full passes over `(pop, 3, ndim)` and one allocation.
  This is now the largest single cost at high dimensions.
- **Set `NUMBA_NUM_THREADS=1`** (or drop `parallel=True` from `hunt`) until the
  per-thread workload is large enough to amortise a parallel region.
- **Avoid the `(pop, 3, ndim)` materialisation entirely** if the algorithm allows
  a leader loop that reuses one coefficient row at a time, keeping the working set
  inside the 256 KiB L2.

## Changes in this branch

- `cmenpy/target/__init__.py`: `VectorizedTarget`, `vectorized_target()`, and
  `TargetFunction.evaluate_matrix()`, plus per-individual nfe accounting.
- `cmenpy/kernel/buffer.py`: `apply()` dispatches vectorized targets to a single
  call, keeping `np.apply_along_axis` as the fallback.
- `cmenpy/__init__.py`: export `vectorized_target` / `VectorizedTarget`.
- `tests/test_target.py`: 11 tests covering dispatch, nfe accounting,
  `MaxNumberOfCalls`, marker strictness, and bit-identical optima.
- `example/benchmark_gwo.py`: rewritten to sweep the `ndim x pop` grid with
  interleaved runs, min/p50 statistics, per-epoch diagnostics, the vectorized A/B
  and the thread sweep.
