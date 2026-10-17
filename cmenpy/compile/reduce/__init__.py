"""The ``cm.reduce`` decorator (numba ``guvectorize``) and the Contract marker."""

from cmenpy.compile.reduce.contract import Contract
from cmenpy.compile.reduce.reduce import reduce

__all__ = ["Contract", "reduce"]
