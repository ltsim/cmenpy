# NFE budget

Every scoring call spends from an evaluation counter — the *number of
function evaluations* (NFE) — and a `TargetFunction` can be given a hard cap
(`max_nfe`) after which it refuses to evaluate anything more.

## Counter semantics

- **Check first, spend after:** `__spend` raises `MaxNumberOfCalls` when
  `nfe + calls > max`, so the counter never exceeds the budget.
- **One individual** (`tf(x)`, `tf.evaluate(x)`) spends `1`. A vectorized
  target gets that solution as a `(1, ndim)` block and the single score is
  unwrapped again, so one row never costs `ndim`.
- **A block** (`tf.evaluate_matrix(x)`) spends `len(x)` — one per row scored.
- **Row-wise fallback** spends `1` per row through the same evaluator.
- **A selection that scores nothing** (an empty mask, every row removed)
  spends `0`: the rows are written, the objective is not called.
- The counter lives on the wrapper, not on your callable, and starts at `0`
  for every wrapper — a fresh `solve` gets a fresh counter.

Spelling the same selection with `pop.compute` costs the same in either
scoring mode — the active rows for `pop.compute << block`, the rows a mask or
index array selects, one row for `pop.compute[i]` (a Python `int` or the
`np.intp` that `pop.best`, `pop.worst` and `pop.sort` hand out).

## Setting a budget

```python
from cmenpy.target import TargetFunction

budgeted = TargetFunction(sphere, bounds, max_nfe=10_000)

budgeted.max          # 10000 (np.inf when max_nfe is None)
budgeted.nfe          # evaluations spent so far
budgeted.f            # bound evaluator (same as evaluate)
budgeted.bounds       # the search space
```

`solve` builds its own internal wrapper **without** a budget, so to cap a run
pass a pre-wrapped target as the objective — the run's scoring routes through
it:

```python
try:
    model.solve(budgeted, bounds, epochs, pop_size)
except MaxNumberOfCalls:
    ...   # budget exhausted mid-run
```

!!! info "Planned for 2026b2"
    Composable terminators — issue
    [#33](https://github.com/ltsim/cmenpy/issues/33) — will add a first-class
    `solve(..., terminator=cm.MaxEvaluation(nfe=...))` alongside time-,
    patience- and target-based criteria. The pre-wrapped `TargetFunction`
    stays the low-level equivalent.

## When the exception fires

`MaxNumberOfCalls` is raised *inside* scoring — during an epoch's
`compute`/`apply` — and propagates out of `solve`. At that point:

- the buffer still holds the last **completed** state (the failing write did
  not finish);
- with `debug=True`, `model.tracker.history` contains every epoch recorded up
  to the failure, and each `EpochHistory` carries its `nfe`;
- the exception itself is a plain `Exception` with the message
  `Max number of function calls exceeded.`

```python
from cmenpy.target import MaxNumberOfCalls

try:
    best = model.solve(budgeted, bounds, epochs, pop_size, debug=True)
except MaxNumberOfCalls as e:
    history = model.tracker.history
    print(f"spent {history[-1].nfe} evaluations over {len(history)} epochs")
    raise
```

## Reading the NFE of a run

- **`EpochHistory.nfe`** (`debug=True`) — the authoritative cumulative count
  at the end of each epoch.
- **`ctx.target.nfe`** — *not* the run's counter. `ctx.target` is a separate
  re-wrap created for the context, so its counter only moves if you call
  through it. Read the run's progress from the tracker instead.

## Sizing a budget

A run scores roughly `pop_size` solutions per epoch (plus whatever your
operators score per epoch), so a first estimate is
`epochs × pop_size` evaluations. Dynamic populations (`pop_range`,
`insert`/`remove`) and multi-evaluation operators add to it — measure once
with `debug=True`, then set the cap with headroom.
