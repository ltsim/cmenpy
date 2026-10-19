"""The `module` decorator: a reusable Operator Module (Ruby-style mixin)."""

import typing

from cmenpy.model.optimizer.variables import install_constants

_ModuleT = typing.TypeVar("_ModuleT")


def module(cls: type[_ModuleT]) -> type[_ModuleT]:
    """Mark ``cls`` as a reusable Operator Module that models can mix in.

    A class decorated with ``@cm.module`` behaves like a Ruby ``module``: it
    groups metaheuristic operators — ``cm.precompile`` / ``cm.vectorize`` /
    ``cm.reduce`` kernels, plain helper methods and ``cm.Argument`` /
    ``cm.Variable`` / ``cm.Constant`` fields — so several ``@cm.template``
    models can inherit the same behavior. Note that ``@cm.module`` returns a
    **class**, not a Python module.

    The decorator keeps the class identity and only marks it as an Operator
    Module and materializes its ``cm.Constant`` annotations, so a constant is
    reachable (``MyOps.P``) even before any model includes the mixin. Mixin
    application is ordinary inheritance; name collisions resolve through the
    normal MRO, with the leftmost base winning.

    Args:
        cls: The Operator Module class to mark.

    Returns:
        The same class, now marked with ``__operator_module__``.

    Raises:
        TypeError: If ``cls`` is not a class (for example a ``@cm.template``
            model instance, which is not reusable as a mixin).

    Examples:
        ::

            @cm.module
            class Selection:
                pressure: cm.Constant[float, 1.5]

                @cm.vectorize
                def clamp(x: cm.Array, lo: cm.Scalar, hi: cm.Scalar):
                    return min(max(x, lo), hi)

            @cm.template
            class GA(Selection):
                def evolve(self, e, ctx) -> None:
                    ...  # can use self.pressure and self.clamp(...)
    """
    if not isinstance(cls, type):
        raise TypeError(f"Operator Module '{cls}' is not a class.")

    install_constants(cls)
    cls.__operator_module__ = True  # type: ignore[attr-defined]

    return cls
