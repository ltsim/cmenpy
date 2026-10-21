# Stream operators

The swarm's verbs are four stream operators. Each is a fresh object bound to
the buffer, and each is used with `<<` (write), `[]` (row select), `~`
(active rows) and, for readers, `>>` (pass-through).

| Operator | Direction | Scores? |
| -------- | --------- | ------- |
| `pop.compute` | write | **yes** — solutions are evaluated against the objective |
| `pop.assign`  | write | no — raw row write, fitness included |
| `pop.access`  | read | — whole rows: fitness + solution |
| `pop.extract` | read | — solution columns only |

## `pop.compute` — write and score

```python
pop.compute << values            # score a whole (k, ndim) block, active rows
pop.compute[i] << row            # score one row into row i
pop.compute[mask] << rows        # score the rows a mask or index array selects
pop.compute[~pop] << rows        # score the rows that hold a fitness
```

`compute` delegates to `KernelBuffer.apply`, which writes the solutions and
fills column 0 with their fitness:

- **no index** → the block is written into the **active** rows
  (`mask.founds`), so inactive (removed) rows are untouched;
- **one row** (`pop.compute[3]`, `pop.compute[pop.best]`) → a single row is
  written and scored with one objective call; `pop.best`, `pop.worst` and
  `pop.sort` hand out `np.intp` indices, which count as one row too;
- **block + vectorized target** → one `evaluate_matrix` call scores the whole
  block ([vectorized targets](../target/vectorized.md));
- **block + plain target** → row-by-row fallback via `apply_along_axis`;
- **an empty selection** (a mask with nothing set) → nothing is evaluated.

Either way the NFE counter is charged **one evaluation per row scored** — the
same selection costs the same with a plain and with a vectorized objective.

The block you write must broadcast into the selected rows' `(k, ndim)` shape.

```python
# Initialize / perturb the whole active population.
pop.compute << ctx.rng.uniform(ctx.bounds.low, ctx.bounds.up,
                               (pop.size, ctx.bounds.ndim))

# Per-row: re-score row i after an in-place tweak.
pop.compute[pop.best] << new_solution

# Only the improved rows: pay for what you actually re-score.
improved = cm.is_best((offspring.F, pop.F), ctx.sense)
pop.compute[improved] << offspring.X[improved]
```

## `pop.assign` — raw write, no scoring

```python
pop.assign << rows      # overwrite the whole buffer (exact shape required)
pop.assign[i] << row    # write one raw row: fitness + solution
```

`assign` writes straight into the buffer without touching the objective:

- the whole-block form raises `IndexError` unless the value's shape is
  exactly the buffer shape `(max, ndim + 1)` — it is meant for restoring
  captured raw rows, not for solutions;
- the per-row form writes a **full row** `(fitness + solution)`. That is the
  tool for merging already-scored candidates (elitist selection) without
  paying for a re-evaluation.

!!! warning "Stale fitness"
    `assign` never validates the fitness against the solution. If you write
    solutions through `assign`, column 0 no longer describes them — use
    `compute` unless you are copying rows that were already scored.

## `pop.access` — whole-row reader

```python
pop.access[i]           # row i: [fitness, x0, x1, ...]
pop.access[i:j]         # rows i..j-1
~pop.access             # whole rows of the active individuals
other >> pop.access     # pass `other` through (identity)
```

Use `access` when a consumer wants the self-contained `(fitness, solution)`
record — for example archiving a row or copying rows between buffers.

## `pop.extract` — solution reader

```python
pop.extract[i]          # solution vector of row i (columns 1:)
pop.extract[i:j]        # solution rows i..j-1
~pop.extract            # solutions of the active individuals
other >> pop.extract    # pass-through
```

`~pop.extract` is the everyday idiom for "the current solutions" — the input
to a mutation step.

## Pipelines

`pop.compute[i]` and `pop.assign[i]` return **pipeline** objects rather than
the operator itself: they are single-use write targets bound to one row, so
the same operator can target any row without re-indexing. A reader's `>>` is
the identity, which lets kernels and helpers be composed in either direction.

## Recipes

### Score candidates in isolation (snapshot)

Evaluate offspring in a fresh buffer that shares the configuration, then
merge only the winners — no re-scoring:

```python
n_pop = cm.PopulationSwarm(buff.snapshot)     # snapshot is a property

n_pop.compute << offspring                     # score in isolation

mask = cm.is_best((n_pop.F, pop.F), ctx.sense)
pop.assign[mask] << n_pop.access[mask]         # copy winning rows, fitness included
```

### Copy selected rows between buffers

```python
buff.raw[idx] = pop.access[j]             # whole row, fitness included
```

### Grow the roster

```python
if pop.free_space:
    pop.insert(candidate)                # scored and inserted; returns the row
```

### Shrink the roster

```python
pop.remove(pop.worst)                     # deactivate + NaN; row reusable
```

`insert`/`remove` and the `pop_range` sizing that enables them are covered in
[Iteration and ranking](ranking.md#dynamic-sizing).
