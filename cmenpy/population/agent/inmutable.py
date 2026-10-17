"""Snapshot agent that copies its buffer row at construction time."""

import typing

import numpy as np

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent.base import BaseAgent


class ImmutableAgent(BaseAgent):
    """A snapshot agent: copies its row out of the buffer at construction.

    Args:
        n: Row index to snapshot.
        buffer: The buffer to read from.
    """

    def __init__(self, n: int, buffer: KernelBuffer) -> None:
        """Copy row ``n`` of ``buffer`` (fitness + solution) into the agent."""
        self.__i = n
        self.__buffer = buffer.raw[n, :].copy()

    def __iter__(self) -> typing.Iterator[np.number[typing.Any]]:
        """Iterate over a copy of the solution coordinates."""
        return iter(self.__buffer[1:].copy())

    def __getitem__(self, key: int) -> np.number[typing.Any]:
        """Return the solution coordinate at ``key``."""
        return typing.cast(np.number[typing.Any], self.__buffer[key + 1])

    @property
    def i(self) -> int:
        """Row index this snapshot was taken from."""
        return self.__i

    @property
    def solution(self) -> NDArrayType:
        """The snapshot solution vector."""
        return self.__buffer[1:]

    @property
    def fitness(self) -> np.number[typing.Any]:
        """The snapshot fitness value."""
        return typing.cast(np.number[typing.Any], self.__buffer[0])

    def __float__(self) -> float:
        """Return the snapshot fitness as a plain float."""
        return float(self.__buffer[0])

    @property
    def x(self) -> NDArrayType:
        """The full snapshot row (fitness in column 0, then the solution)."""
        return self.__buffer
