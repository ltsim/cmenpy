# Target functions

`TargetFunction` wraps the objective and counts function evaluations (NFE), with
an optional `max_nfe` budget that raises `MaxNumberOfCalls` when exceeded. Plain
targets are scored row-by-row; a vectorized target scores the whole
`(pop, ndim)` matrix in one call:

```python
def sphere_block(X):              # X: (pop, ndim) -> (pop,)
    return np.sum(X**2, axis=1)

model.solve(cm.vectorized_target(sphere_block), bounds, epochs, pop_size)
```

The `vectorized_target` marker must be exactly `True` on the wrapper (a truthy
string does not count); anything else falls back to `np.apply_along_axis`.

## API

::: cmenpy.vectorized_target
::: cmenpy.VectorizedTarget
