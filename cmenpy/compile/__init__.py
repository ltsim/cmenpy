"""Numba kernel decorators, their typing markers and introspection helpers."""

from cmenpy.compile.kernel import _Kernel as _Kernel
from cmenpy.compile.precompile import jit, precompile
from cmenpy.compile.prepare import kernels, prepare
from cmenpy.compile.reduce import Contract, reduce
from cmenpy.compile.vectorize import vectorize
from cmenpy.hints.spec import Array, Scalar

__all__ = [
    "Array",
    "Contract",
    "Scalar",
    "jit",
    "kernels",
    "precompile",
    "prepare",
    "reduce",
    "vectorize",
]
