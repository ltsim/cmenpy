"""Numba kernel decorators, their typing markers and introspection helpers."""

import typing

from cmenpy.compile import backend as _backend
from cmenpy.compile.kernel import _Kernel as _Kernel
from cmenpy.compile.precompile import jit, precompile
from cmenpy.compile.prepare import kernels, prepare
from cmenpy.compile.reduce import Contract, reduce
from cmenpy.compile.vectorize import vectorize
from cmenpy.hints.spec import Array, Scalar

__all__ = [
    "HAS_NUMBA",
    "Array",
    "Contract",
    "Scalar",
    "jit",
    "kernels",
    "prange",
    "precompile",
    "prepare",
    "reduce",
    "vectorize",
]


def __getattr__(name: str) -> typing.Any:
    """Expose ``prange``/``HAS_NUMBA`` lazily from the optional-numba backend."""
    try:
        return _backend.lazy(name)
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
