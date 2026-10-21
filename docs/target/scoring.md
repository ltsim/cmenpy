# Scoring modes

There are two ways a target is scored, and the difference is the single most
important performance lever in `cmenpy`:

| Mode | How it scores | NFE spent | When |
| ---- | ------------- | --------- | ---- |
| **Row-by-row** (default) | `np.apply_along_axis` — one Python call per individual | 1 per individual | a plain callable |
| **Vectorized** (fast) | one call over the whole `(pop, ndim)` block | `len(block)` | wrapped with `cm.vectorized_target` |

## Row-by-row

A plain Python callable works as-is:

```python
def sphere(x):                      # x: one solution vector -> one float
    return float(np.sum(x**2))

model.solve(sphere, bounds, epochs, pop_size)     # scored row-by-row
```

Each epoch, `KernelBuffer.apply` feeds the active rows through
`np.apply_along_axis(self.__target, 1, ...)`: one Python-level call per
individual, one NFE each. This is fine for prototypes and cheap objectives;
for anything expensive it dominates the runtime.

## Vectorized block

A vectorized target receives the **entire active block at once** and returns
one fitness per row:

```python
def sphere_block(x):                # x: (pop, ndim) -> (pop,)
    return np.sum(x**2, axis=1)

model.solve(cm.vectorized_target(sphere_block), bounds, epochs, pop_size)
```

One call per epoch instead of one per individual — typically an order of
magnitude faster for NumPy-shaped objectives. The contract and pitfalls are
on [Vectorized targets](vectorized.md).

## How dispatch works

`KernelBuffer.apply(x, idx)` picks the path from the index and the marker:

| Call | Path | NFE |
| ---- | ---- | --- |
| `apply(x, idx=int)` | `target.evaluate(x)` — one call | 1 |
| `apply(x, idx=block)` with `target.vectorized` | `target.evaluate_matrix(block)` | `len(block)` |
| `apply(x, idx=block or None)`, plain target | `np.apply_along_axis(...)` over the rows | 1 per row |

With no index, `apply` writes and scores the **active** rows only
(`mask.founds`), so removed individuals are never re-scored and never spend
budget.

## Equivalence

The two modes produce identical fitness — only the dispatch differs:

```python
x = np.array([[1.0, 2.0], [-3.0, 0.5], [0.0, 0.0]])

rowwise   = [sphere(row) for row in x]
block     = TargetFunction(cm.vectorized_target(sphere_block), BOUNDS)

np.testing.assert_allclose(rowwise, block.evaluate_matrix(x))
```

## `apply_axis`

`TargetFunction.apply_axis(x, mask)` scores the rows selected by an integer
`mask`, using `apply_along_axis` internally. Note that it expects the full
raw rows (fitness in column 0) and drops column 0 before calling the
objective — it is the low-level helper behind row-wise evaluation.

## Correctness checklist for a block objective

- **Shape in, shape out:** returns exactly `(len(x),)` — one value per row.
- **Axis:** reductions use `axis=1` (per row), never the default whole-array
  sum.
- **dtype:** float in, float out; `float(...)` a scalar result per row.
- **No in-place writes** to `x` — it is the live buffer's solution block.
- **Equivalence:** assert your block version against the row-wise version on
  a small block before trusting it at scale.
