# Population DSL

`cm.PopulationSwarm(buffer)` is the main handle over a `KernelBuffer` — a
`(pop_size, ndim+1)` store with fitness in column 0 and the solution vector from
column 1.

```python
pop = cm.PopulationSwarm(ctx.buff)     # pop.X = solutions, pop.F = fitness
pop.compute << values                  # write a whole (pop_size, ndim) block and score it
pop.assign[mask] << pop.access[mask]   # copy rows (access = read-only, assign = writable)
~pop.extract                           # active/found solutions; ~pop.idx / ~pop.F filter active rows
```

- Iterate with `for i in pop:` or `for i in ~pop:`.
- Selection and ranking follow `ctx.sense` (`"min"`/`"max"`): `pop.best`,
  `pop.worst`, `pop.sort`, plus `cm.best_index` / `cm.worst_index` and the
  `cm.is_best((new_F, old_F), ctx.sense)` comparator.
- Manage the roster with `pop.size`, `pop.insert(x)`, `pop.remove(i)`,
  `pop.free_space`.
- The usual offspring pattern scores candidates in a fresh buffer and merges:
  `ctx.buff.snapshot` → score → merge with `cm.is_best`.

## API

::: cmenpy.PopulationSwarm
::: cmenpy.KernelBuffer
::: cmenpy.KernelSize
::: cmenpy.is_best
::: cmenpy.is_worst
::: cmenpy.best_index
::: cmenpy.worst_index
::: cmenpy.Bounds
