"""Compute Metaheuristic Engine for Python (cmenpy)."""

__version__ = "2026b1"

import typing

from cmenpy.bounds import Bounds
from cmenpy.compile import (
    Array,
    Contract,
    Scalar,
    jit,
    precompile,
    prepare,
    reduce,
    vectorize,
)
from cmenpy.compile import backend as _backend
from cmenpy.epoch import EpochIteration
from cmenpy.functools import best_index, is_best, is_worst, worst_index
from cmenpy.kernel import KernelBuffer, KernelSize
from cmenpy.mixin import module
from cmenpy.model import declare, template
from cmenpy.model.optimizer import FunctionOptimizerModel, TemplateOptimizerModel
from cmenpy.model.optimizer.variables import Argument, Constant, Variable
from cmenpy.population.swarm import PopulationSwarm
from cmenpy.target import VectorizedTarget, vectorized_target

__all__ = [
    "HAS_NUMBA",
    "Argument",
    "Array",
    "Bounds",
    "Constant",
    "Contract",
    "EpochIteration",
    "FunctionOptimizerModel",
    "KernelBuffer",
    "KernelSize",
    "PopulationSwarm",
    "Scalar",
    "TemplateOptimizerModel",
    "Variable",
    "VectorizedTarget",
    "best_index",
    "declare",
    "is_best",
    "is_worst",
    "jit",
    "module",
    "prange",
    "precompile",
    "prepare",
    "reduce",
    "template",
    "vectorize",
    "vectorized_target",
    "worst_index",
]


def __getattr__(name: str) -> typing.Any:
    """Expose ``prange``/``HAS_NUMBA`` lazily from the optional-numba backend.

    ``from cmenpy import prange`` binds numba's ``prange`` when numba is
    installed and the builtin ``range`` otherwise, so kernels written with it
    work in both modes.
    """
    if name == "prange":
        return _backend.prange()

    if name == "HAS_NUMBA":
        return _backend.HAS_NUMBA

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
