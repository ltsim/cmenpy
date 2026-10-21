# Functions

Every decorator and free function the framework exposes, grouped by role.
Classes live in [Classes](classes.md); the packages themselves are described
in [Modules](modules.md).

## Declaration decorators

The two ways to declare a model (`declare` function-style, `template`
class-style), the Operator Module decorator, and the vectorized-objective
decorator.

::: cmenpy.declare
    options:
      heading_level: 3

::: cmenpy.template
    options:
      heading_level: 3

::: cmenpy.module
    options:
      heading_level: 3

::: cmenpy.vectorized_target
    options:
      heading_level: 3

## Kernel decorators

Numba kernel decorators, re-exported at the top level, so `cmenpy.precompile`
and `cmenpy.compile.precompile` are the same object.

::: cmenpy.precompile
    options:
      heading_level: 3

::: cmenpy.vectorize
    options:
      heading_level: 3

::: cmenpy.reduce
    options:
      heading_level: 3

::: cmenpy.jit
    options:
      heading_level: 3

## Signature markers

Typing markers that appear in kernel signatures: the `Array`/`Scalar` spec
markers and the `Contract` signature marker for `reduce`.

::: cmenpy.Array
    options:
      heading_level: 3

::: cmenpy.Scalar
    options:
      heading_level: 3

::: cmenpy.Contract
    options:
      heading_level: 3

## Builders and factories

The helpers that turn a declaration into a runnable model, and the small
factories for bounds and buffers.

::: cmenpy.model.declare_from_func
    options:
      heading_level: 3

::: cmenpy.model.declare_from_class
    options:
      heading_level: 3

::: cmenpy.model.optimizer.base.build_context_manager
    options:
      heading_level: 3

::: cmenpy.bounds.create_bounds
    options:
      heading_level: 3

::: cmenpy.kernel.low.init_buffer
    options:
      heading_level: 3

## Introspection

::: cmenpy.compile.kernels
    options:
      heading_level: 3

::: cmenpy.prepare
    options:
      heading_level: 3

## Comparators

Sense-aware fitness comparators used by selection operators.

::: cmenpy.is_best
    options:
      heading_level: 3

::: cmenpy.is_worst
    options:
      heading_level: 3

::: cmenpy.best_index
    options:
      heading_level: 3

::: cmenpy.worst_index
    options:
      heading_level: 3
