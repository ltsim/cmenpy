# Iteration and ranking

All ordering in the population is **sense-aware**: with `ctx.sense = "min"`
(smallest fitness first, the default) the lowest fitness is *best*; with
`"max"` the highest is. Take the sense from `ctx.sense` and let the ranking
API apply it — never hardcode a direction.

## Iterating

```python
for i in pop:            # every row index of the buffer, 0 .. max-1
    ...

for i in ~pop:           # only rows with a scored (non-NaN) fitness
    ...
```

`~pop` yields the indices of rows whose fitness is **not NaN** — that is,
rows that have been scored. A freshly created buffer scores its active rows
during initialization, so in practice `~pop` and the active roster coincide;
the difference appears after `remove`, where the row becomes NaN *and*
inactive.

To iterate the data instead of the indices, use the views:

```python
for x in ~pop.extract:       # one solution vector at a time (copies)
    ...

from cmenpy.population.agent import ImmutableAgent
agent = ImmutableAgent(pop.best, ctx.buff)   # object view of one row
```

## `best`, `worst`, `sort`

```python
pop.best          # row index of the best individual
pop.worst         # row index of the worst individual
pop.sort          # row indices ordered best-first

pop.F[pop.best]   # the best fitness value
pop.X[pop.sort[:k]]   # the k best solution vectors
```

These delegate to the buffer's `BufferIndex`, which ranks over the fitness
column only. Notes:

- **Ties** keep the earlier (lower) row first — the ordering is stable.
- **NaN rows** sort last in `sort` and are never `best`/`worst` while any
  scored row exists; on an all-NaN buffer the ranking is meaningless, so
  initialize before reading it.

`pop.idx` exposes the same lookups (`pop.idx.sort`, `pop.idx.best`,
`pop.idx.worst`, `~pop.idx`) plus iteration over the sorted order.

## Comparators

For element-wise comparisons between two fitness arrays — offspring vs
parents, candidates vs personal bests — use the comparators instead of
rolling your own:

```python
improved = cm.is_best((new_F, old_F), ctx.sense)     # indices where new wins
worse    = cm.is_worst((new_F, old_F), ctx.sense)    # indices where new loses

best_i   = cm.best_index(F, ctx.sense)   # index of the best entry of F
worst_i  = cm.worst_index(F, ctx.sense)  # index of the worst entry of F
```

- `is_best`/`is_worst` take a `(new, old)` pair — a tuple of scalars or
  arrays — and return the **integer indices** where the comparison holds.
  NaN entries never compare, so masked/inactive rows are skipped naturally.
- `is_best(a, sense) == ~is_worst(a, sense)` for scored data.
- `best_index`/`worst_index` operate on a single fitness array.

Full reference:
[API Reference → Comparators](../api/functions.md#comparators).

## Dynamic sizing

A run can grow and shrink its roster within the bounds given to `solve`:

```python
model.solve(f, bounds, epochs, pop_size=20, pop_range=(5, 50))
```

`pop_size` is the initial size; `pop_range` sets the minimum and maximum the
roster may move between. Within those limits:

```python
pop.size                 # current active individuals
pop.free_space           # True while insert() would succeed

pop.remove(pop.worst)    # deactivate a row: mask off + NaN
idx = pop.insert(x)      # score x into the first free row; returns the row
```

`insert` raises `ValueError("Max agents in memory.")` when no row is free, so
guard growth with `pop.free_space`. The buffer never reallocates: capacity is
`ctx.buff.size.max`.

The classic pattern — remove the worst when a new solution beats it, insert a
replacement otherwise — is `example/hello_world_v1.py`; `example/optimizers/de/lshade_algorithm.py`
shows success-history adaptation over a shrinking roster.
