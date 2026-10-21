# Population

`cm.PopulationSwarm(buffer)` is the handle you write algorithm bodies with.
Underneath it is a `KernelBuffer`: one flat `(pop_size, ndim + 1)` array where
**column 0 is the fitness** and **columns `1:` are the solution vector** — one
row per individual, laid out the way Numba kernels want it.

```python
pop = cm.PopulationSwarm(ctx.buff)   # pop.X = solutions, pop.F = fitness
```

## What the swarm composes

`PopulationSwarm` is four mixins over the same buffer:

| Mixin | Provides |
| ----- | -------- |
| `StreamOperations` | the DSL verbs: `compute`, `assign`, `access`, `extract` |
| `BufferMatrixOperations` | the views `X`, `F`, `idx`, `size` |
| `BaseIterator` | `for i in pop`, `best`, `worst`, `sort` |
| `DynamicBase` | roster: `insert`, `remove`, `free_space` |

## The buffer model

- The buffer is allocated to the **maximum** population size (`pop_range`);
  `pop.size` is the number of *active* rows.
- A row is active while it holds a live individual; `pop.remove(i)` deactivates
  it and fills it with `NaN`, and `pop.insert(x)` reuses the first free row.
- Fitness lives in column 0, so every row is a self-contained `(fitness,
  solution)` record — that is what makes whole-row copies and block scoring
  cheap.

## 60-second tour

```python
pop = cm.PopulationSwarm(ctx.buff)

# Initialize: write a whole (pop_size, ndim) block and score it.
pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                               (pop.size, ctx.bounds.ndim))

best = pop.F[pop.best]                       # fitness of the best individual

for i in pop:                                # every row index
    ...

for _ in epoch:
    # Score offspring: ~pop.X / ~extract are the active solution rows.
    pop.compute << np.clip(~pop.X + ctx.rng.uniform(-1, 1,
                                                    (pop.size, ctx.bounds.ndim)),
                           ctx.bounds.low, ctx.bounds.up)

    if pop.F[pop.best] < best:               # sense-aware comparison
        pop.remove(pop.worst)                # shrink the roster
    elif pop.free_space:
        pop.insert(ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                                   ctx.bounds.ndim))   # grow it
```

## I want to…

| Task | Use |
| ---- | --- |
| write solutions **and** score them | [`pop.compute << values`](operators.md) |
| write rows without scoring (raw) | `pop.assign << values` |
| read the solution block | `pop.X`, `~pop.X`, `pop.extract` |
| read whole rows (fitness + solution) | `pop.access[i]`, `~pop.access` |
| read the fitness column | `pop.F`, `~pop.F` |
| find best / worst / rank | [`best`, `worst`, `sort`](ranking.md) |
| iterate individuals | `for i in pop` / `for i in ~pop` |
| grow or shrink the roster | `insert` / `remove` / `free_space` |
| score candidates in isolation | `ctx.buff.snapshot` + `assign` |

## Where to go next

- [Matrix views](views.md) — `X`, `F`, `idx`, `size`, live views and copies.
- [Stream operators](operators.md) — the `<<` / `[]` / `~` DSL and recipes.
- [Iteration and ranking](ranking.md) — ordering, comparators, dynamic sizing.

Full class reference:
[API Reference → Classes](../api/classes.md#population); buffer types under
[Buffer](../api/classes.md#buffer), comparators under
[Comparators](../api/functions.md#comparators), `Bounds` under
[Search space](../api/classes.md#search-space).
