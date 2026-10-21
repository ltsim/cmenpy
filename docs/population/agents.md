# Agents

An **agent** is a Python object bound to one buffer row — the object-oriented
counterpart of the array-first DSL. The algorithm internals never use them
(they stay vectorized), but agents are convenient for archives, reporting and
small per-individual tweaks.

```python
from cmenpy.population.agent import ImmutableAgent, MutableAgent
```

Agents are not part of the top-level `cm` namespace; import them from
`cmenpy.population.agent`.

## The `BaseAgent` contract

Both variants expose the same surface:

| Member | Meaning |
| ------ | ------- |
| `agent.i` | row index in the buffer |
| `agent.solution` | the solution vector (columns `1:`) |
| `agent.fitness` | the fitness value (column 0) |
| `agent.x` | the full row: `[fitness, x0, x1, ...]` |
| `float(agent)` | the fitness as a plain `float` |
| `agent[k]` | the `k`-th solution coordinate |
| `for v in agent` | iterate the solution coordinates |
| `agent < other`, `agent > other` | compare by fitness (against another agent or a scalar) |
| `hash(agent)` | hashes by row index |

Comparison is by fitness and **not** sense-aware — `a < b` literally means
`a.fitness < b.fitness`, which is "better" only under `sense="min"`. For
sense-aware decisions use the [comparators](ranking.md#comparators).

## `ImmutableAgent` — a snapshot

```python
best = ImmutableAgent(pop.best, ctx.buff)

best.i            # the row it was taken from
best.solution     # a copy of the row's solutions at construction time
float(best)       # the fitness at construction time
```

The constructor **copies** the row. Later writes to the buffer — including
re-scoring that row — do not affect the agent, so it is the right tool for
archives and for keeping a historical best across epochs. It is read-only by
construction: there is no `solution` setter.

## `MutableAgent` — a live row

```python
a = MutableAgent(pop.best, ctx.buff)

a.solution                 # a live view of the row's solutions
a.solution = new_x         # writes the row AND re-scores it (an NFE is spent)
```

`MutableAgent` holds the row index, not a copy: every read reflects the
buffer's current state, and assigning `solution` routes through
`buffer.apply`, so the fitness column is recomputed through the objective.
Notes:

- each `solution = ...` assignment costs one function evaluation;
- the agent follows its row — after `pop.remove(i)` the agent reads NaN;
- mutating `a.solution` in place (`a.solution[0] = ...`) changes the buffer
  but does **not** re-score; assign a whole new vector when the fitness must
  be updated.

## When to use which

| Situation | Use |
| --------- | --- |
| archive the best of a run; report an individual | `ImmutableAgent` |
| keep a snapshot across epochs while the buffer changes | `ImmutableAgent` |
| stepwise in-place updates with automatic re-scoring | `MutableAgent` |
| read-only inspection of a live row | `MutableAgent` |
| anything inside the epoch loop over many individuals | arrays and the [operators](operators.md) — an object per individual is far slower |

## Recipe: keep a historical best

```python
from cmenpy.population.agent import ImmutableAgent

history_best = None

for _ in epoch:
    ...
    if history_best is None or pop.F[pop.best] < float(history_best):
        history_best = ImmutableAgent(pop.best, ctx.buff)

print(float(history_best), history_best.solution)
```
