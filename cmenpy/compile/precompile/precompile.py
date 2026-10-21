"""The ``cm.precompile`` decorator (``numba.njit`` with spec/annotation support)."""

import typing

from cmenpy.compile.backend import numba_module
from cmenpy.compile.low.decorate import _template


def _njit(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Resolve ``numba.njit`` on use, so declaring needs no numba import."""
    return numba_module().njit(*args, **kwargs)


def precompile(
    *args: typing.Any,
    fastmath: bool = False,
    parallel: bool = False,
    **kwargs: typing.Any,
) -> typing.Any:
    """Precompile a static method with `numba.njit`.

    Specs can be passed positionally (``cm.Array``, ``cm.Array[float, 2]``) or
    declared as annotations in the decorated function::

        @cm.precompile
        def hunt(X: "Array[float, 2]", a: "Scalar[float]"): ...

    ``fastmath`` and ``parallel`` map to the `numba.njit` options of the
    same name (``parallel`` also needs `numba.prange` in the body). A model
    can also enable them for every kernel at instantiation. Any other numba
    option (``cache=True``, ``boundscheck=False``, ...) is forwarded through
    keyword arguments. Pass ``numba=False`` to force this kernel to run as
    plain Python even when numba is available.
    """
    return _template(
        _njit,
        parallel_mode="njit",
        parallel=parallel,
        fastmath=fastmath,
        **kwargs,
    )(*args)
