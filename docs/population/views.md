# Matrix views

The swarm exposes the buffer through four views. They are *views*: reads and
writes go straight to the backing array and take effect immediately.

| View | Type | Contents |
| ---- | ---- | -------- |
| `pop.X` | read/write | the solution block, shape `(max, ndim)` (buffer columns `1:`) |
| `pop.F` | read-only | the fitness column, shape `(max,)` (buffer column 0) |
| `pop.idx` | lookups | best/worst/sort over the fitness column |
| `pop.size` | `int` | number of **active** individuals |

## Reading

```python
pop.X[i]          # solution vector of row i
pop.X[i:j]        # solution rows i..j-1
pop.F[i]          # fitness of row i
np.asarray(pop.X) # the whole (max, ndim) block, as an ndarray
np.asarray(pop.F) # the whole fitness column
```

`~` restricts a view to the **active** rows:

```python
~pop.X            # (active, ndim) solutions of live individuals
~pop.F            # fitness of live individuals
```

## Writing

`pop.X` is writable through the scoring operator:

```python
pop.X << values       # score `values` and insert it — same as pop.compute << values
```

`pop.X` and `pop.F` are live views. Read the whole block with `pop.X.to_numpy()`
(or `cm.cast(pop.X)`) — implicit `np.asarray(pop.X)` casting is deprecated.
They can also be passed directly to `cm.precompile`/`cm.vectorize`/`cm.reduce`
kernels (cmenpy unwraps them internally).

`pop.F` has no write path: fitness is produced by scoring
([`compute`](operators.md)), never assigned directly. To write a raw row
(fitness *and* solution) without scoring, use `pop.assign`.

!!! warning "`np.asarray` does not copy"
    `np.asarray(pop.X)` and `np.asarray(pop.F)` return the backing arrays
    themselves. Mutating the result mutates the population — including its
    fitness column, which then no longer matches the solutions. Take a copy
    (`.copy()`) when you need an isolated array.

## Live views vs snapshots

```python
pop.X[i]                  # live: reflects every write made since
snap = ctx.buff.snapshot  # a fresh, empty buffer with the same configuration
```

`snapshot` is a property returning a new empty `KernelBuffer`, useful for
scoring candidate offspring in isolation before merging them (see
[operators → recipes](operators.md#recipes)).

## `pop.idx` — ranked lookups

```python
pop.idx.sort      # row indices ordered best-first (respects ctx.sense)
pop.idx.best      # row index of the best individual
pop.idx.worst     # row index of the worst individual
~pop.idx          # row indices whose fitness is scored (not NaN)
pop.idx[i]        # i-th best row index (iteration is over sort)
```

All ranking is **sense-aware**: with `sense="min"` the lowest fitness is
"best"; with `sense="max"` the highest is. Take the sense from `ctx.sense` —
never hardcode a direction. See [Iteration and ranking](ranking.md).

## `pop.size` vs capacity

```python
pop.size              # active individuals (int)
ctx.buff.size.max     # capacity: rows the buffer was allocated for
ctx.buff.size.min     # lower bound configured via pop_range
pop.free_space        # True while insert() would succeed
```

`pop.size` changes as rows are removed and inserted; the buffer itself never
resizes — dynamic populations use `pop_range` at `solve` time and
`remove`/`insert` to move within it. See
[Iteration and ranking → dynamic sizing](ranking.md#dynamic-sizing).
