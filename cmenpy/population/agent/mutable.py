"""Live agent bound to a buffer row; writes propagate back."""

import typing

import numpy as np

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent.base import BaseAgent


class MutableAgent(BaseAgent):
    """A live agent bound to a buffer row; writes propagate back.

    Assigning `solution` re-evaluates the row through the buffer.

    Args:
        n: Row index to bind to.
        buffer: The buffer to read from and write to.
    """

    def __init__(self, n: int, buffer: KernelBuffer) -> None:
        """Bind the agent to row ``n`` of ``buffer``."""
        self.__i = n
        self.__buffer = buffer

    def __iter__(self) -> typing.Iterator[np.number[typing.Any]]:
        """Iterate over a copy of the row's solution coordinates."""
        return iter(self.__buffer.raw[self.__i, 1:].copy())

    def __getitem__(self, key: int) -> np.number[typing.Any]:
        """Return the solution coordinate at ``key``."""
        return typing.cast(np.number[typing.Any], self.__buffer.raw[self.__i, key + 1])

    @property
    def i(self) -> int:
        """Row index this agent is bound to."""
        return self.__i

    @property
    def solution(self) -> NDArrayType:
        """The row's live solution vector."""
        return self.__buffer.raw[self.i, 1:]

    @solution.setter
    def solution(self, value: NDArrayType) -> None:
        """Write a new solution into the row and re-evaluate its fitness."""
        self.__buffer.apply(value, self.__i)

    @property
    def fitness(self) -> np.number[typing.Any]:
        """The row's live fitness value."""
        return typing.cast(np.number[typing.Any], self.__buffer.raw[self.i, 0])

    def __float__(self) -> float:
        """Return the live fitness as a plain float."""
        return float(self.__buffer.raw[self.i, 0])

    @property
    def x(self) -> NDArrayType:
        """The full live row (fitness in column 0, then the solution)."""
        return self.__buffer.raw[self.i, :]
