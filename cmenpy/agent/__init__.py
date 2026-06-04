import abc
import typing

import numpy as np

from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class Agent(abc.ABC):
    def __lt__(self, other):
        if isinstance(other, Agent):
            return self.fitness < other.fitness
        elif any(isinstance(other, t) for t in (int, float)):
            return self.fitness < other

        return False

    def __gt__(self, other):
        if isinstance(other, Agent):
            return self.fitness > other.fitness
        elif any(isinstance(other, t) for t in (int, float)):
            return self.fitness > other

        return False

    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __getitem__(self, key: int) -> float: ...

    @property
    @abc.abstractmethod
    def id(self) -> int: ...

    @property
    @abc.abstractmethod
    def solution(self): ...

    @property
    @abc.abstractmethod
    def fitness(self) -> int | float: ...

    @abc.abstractmethod
    def __float__(self) -> float: ...

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, fitness={self.fitness}, solution={self.solution})"

    def __hash__(self):
        return hash(self.id)

    @property
    @abc.abstractmethod
    def buff(self) -> NDArrayType: ...


class MemoryAgent(Agent):
    def __init__(self, buffer: np.ndarray, n: int, target: TargetFunction):
        self.__buffer = buffer
        self.__id = n
        self.__target = target

    def __iter__(self):
        return iter(self.__buffer.copy())

    def __getitem__(self, key: int) -> float:
        return self.__buffer[key]

    @property
    def id(self):
        return self.__id

    @property
    def solution(self):
        return self.__buffer[self.id, 1:]

    @solution.setter
    def solution(self, value: NDArrayType):
        self.__buffer[self.id] = self.__target.evaluate(value)

    @property
    def fitness(self):
        return self.__buffer[self.id, 0]

    def __float__(self):
        return self.__buffer[self.id, 0]

    @property
    def buff(self) -> NDArrayType:
        return self.__buffer


class VirtualAgent(Agent):
    def __init__(self, buffer: np.ndarray, n: int):
        self.__buffer = buffer[n, :].copy()
        self.__id = n

    def __iter__(self):
        return iter(self.__buffer.copy())

    def __getitem__(self, key: int) -> float:
        return self.__buffer[key]

    @property
    def id(self):
        return self.__id

    @property
    def solution(self):
        return self.__buffer[1:]

    @property
    def fitness(self):
        return self.__buffer[0]

    def __float__(self):
        return self.__buffer[0]

    @property
    def buff(self) -> NDArrayType:
        return self.__buffer


class AgentTemplate(typing.Protocol):
    id: int
    solution: np.ndarray
    fitness: float
