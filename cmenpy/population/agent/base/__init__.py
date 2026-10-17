"""Abstract base for population agents (identity, fitness, ordering)."""

import abc
import typing

import numpy as np

from cmenpy.hints import NDArrayType, ScalarType


class BaseAgent(abc.ABC):
    """Abstract individual: fitness, solution vector and ordering.

    Agents compare by fitness (``<``/``>`` against another agent or a
    scalar), so populations can be ranked element-wise.
    """

    def __lt__(self, other: typing.Union["BaseAgent", ScalarType]) -> bool:
        """Return whether this agent's fitness is lower than ``other``."""
        if isinstance(other, BaseAgent):
            return bool(self.fitness < other.fitness)
        elif any(isinstance(other, t) for t in (int, float)):
            return bool(self.fitness < other)

        return False

    def __gt__(self, other: typing.Union["BaseAgent", ScalarType]) -> bool:
        """Return whether this agent's fitness is greater than ``other``."""
        if isinstance(other, BaseAgent):
            return bool(self.fitness > other.fitness)
        elif any(isinstance(other, t) for t in (int, float)):
            return bool(self.fitness > other)

        return False

    @abc.abstractmethod
    def __iter__(self) -> typing.Iterator[np.number[typing.Any]]:
        """Iterate over the solution coordinates."""

    @abc.abstractmethod
    def __getitem__(self, key: int) -> np.number[typing.Any]:
        """Return the solution coordinate at ``key``."""

    @property
    @abc.abstractmethod
    def i(self) -> int:
        """Row index of the agent in the buffer."""

    @property
    @abc.abstractmethod
    def solution(self) -> NDArrayType:
        """The agent's solution vector."""

    @property
    @abc.abstractmethod
    def fitness(self) -> np.number[typing.Any]:
        """The agent's fitness value."""

    @property
    @abc.abstractmethod
    def x(self) -> NDArrayType:
        """The agent's full row (fitness in column 0, then the solution)."""

    @abc.abstractmethod
    def __float__(self) -> float:
        """Return the fitness as a plain float."""

    def __repr__(self) -> str:
        """Return a debug representation of the agent."""
        return f"Agent(i={self.i}, fitness={self.fitness}, solution={self.solution})"

    def __hash__(self) -> int:
        """Hash by the agent's identity index."""
        return hash(self.i)
