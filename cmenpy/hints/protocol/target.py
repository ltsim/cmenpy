"""Target protocol: the signature of an objective callable."""

import typing

from cmenpy.hints.array import NDArrayType


class Target(typing.Protocol):
    """Protocol for an objective: a callable mapping a solution to a score."""

    def __call__(self, x: NDArrayType) -> NDArrayType | typing.Any:
        """Score a solution vector (or matrix of solutions)."""
