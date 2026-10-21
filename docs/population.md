# Population

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
- `pop.X` and `pop.F` are live views over the buffer. Read the whole block with
  `pop.X.to_numpy()` (or `cm.cast(pop.X)`); a slice such as `pop.X[:n]` is
  already an ndarray. Implicit `np.asarray(pop.X)` casting is deprecated. These
  views can also be passed directly to `cm.precompile`/`cm.vectorize`/`cm.reduce`
  kernels — `cmenpy` unwraps them internally.
- Selection and ranking follow `ctx.sense` (`"min"`/`"max"`): `pop.best`,
  `pop.worst`, `pop.sort`, plus `cm.best_index` / `cm.worst_index` and the
  `cm.is_best((new_F, old_F), ctx.sense)` comparator.
- Manage the roster with `pop.size`, `pop.insert(x)`, `pop.remove(i)`,
  `pop.free_space`.
- The usual offspring pattern scores candidates in a fresh buffer and merges:
  `ctx.buff.snapshot` → score → merge with `cm.is_best`.

## API

Autodoc for the swarm handle, operators, matrix views and agents lives in the
[API Reference → Population](api/population.md). The buffer types are under
[Kernel](api/kernel.md), the comparators under [Utilities](api/utilities.md),
and `Bounds` under [Bounds](api/bounds.md).
