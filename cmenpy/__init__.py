"""Compute Metaheuristic Engine for Python (cmenpy)."""

__version__ = "2026b1"

from cmenpy.bounds import Bounds
from cmenpy.casting import cast
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
    "cast",
    "declare",
    "is_best",
    "is_worst",
    "jit",
    "module",
    "precompile",
    "prepare",
    "reduce",
    "template",
    "vectorize",
    "vectorized_target",
    "worst_index",
]
