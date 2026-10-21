# Target functions

A *target* is the objective you are optimizing. `cmenpy` accepts any callable
that maps one solution vector to one fitness value — a Python function, a
compiled kernel, or a whole-block evaluator — and wraps it in a
`TargetFunction` that counts function evaluations (NFE) and enforces an
optional budget.

You normally never build the wrapper yourself: `solve` wraps whatever you
pass as the first argument.

```python
model.solve(lambda x: float(np.sum(x**2)), bounds, epochs=100, pop_size=20)
```

## The scoring pipeline

Every solution enters the population through the buffer, and the buffer owns
the scoring:

```
new solutions  →  KernelBuffer.apply  →  write column 1:  →  score  →  column 0
                                                  │
                    ┌─────────────────────────────┼──────────────────────────┐
                    │                             │                          │
            single row (idx int)           vectorized block            plain callable
            one target( ) call             evaluate( ) once            apply_along_axis
            nfe += 1                       nfe += len(block)           nfe += 1 per row
```

The three paths are what [Scoring modes](scoring.md) covers; the middle one
is the single biggest performance lever in the library.

## `TargetFunction` surface

| Member | Meaning |
| ------ | ------- |
| `tf(x)` / `tf.evaluate(x)` | score row-by-row, spending the budget |
| `tf.evaluate_matrix(x)` | score a whole `(pop, ndim)` block (vectorized only) |
| `tf.nfe` | evaluations spent so far |
| `tf.max` | the budget (`inf` when unlimited) |
| `tf.bounds` | the search space |
| `tf.vectorized` | whether the target scores whole blocks |

The `Target` protocol itself is structural: anything callable with
`x -> float` qualifies.

## Choosing an evaluation mode

| You have… | Mode | Page |
| --------- | ---- | ---- |
| a quick prototype, a cheap objective | plain callable, row-by-row | [Scoring modes](scoring.md) |
| a NumPy-level `(pop, ndim) -> (pop,)` objective | vectorized | [Vectorized targets](vectorized.md) |
| a Numba-compiled block kernel | vectorized + `@cm.precompile` | [Vectorized targets](vectorized.md) |
| several objectives at once | multiobjective | planned for 2026b2 ([#36](https://github.com/ltsim/cmenpy/issues/36)) |
| a hard cap on evaluations | budget | [NFE budget](budget.md) |

## Where to go next

- [Scoring modes](scoring.md) — row-by-row vs whole-block dispatch.
- [Vectorized targets](vectorized.md) — the contract, examples, pitfalls.
- [NFE budget](budget.md) — `max_nfe`, `MaxNumberOfCalls`, accounting.

Full reference: [API Reference → Classes](../api/classes.md) and runnable
targets in [Examples](../examples.md).
