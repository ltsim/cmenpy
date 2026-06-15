import abc
import typing

from cmenpy.hints import ArrayType, ScalarType


class BaseAgent(abc.ABC):
    def __lt__(self, other: typing.Union["BaseAgent", ScalarType]):
        if isinstance(other, BaseAgent):
            return self.fitness < other.fitness
        elif any(isinstance(other, t) for t in (int, float)):
            return self.fitness < other

        return False

    def __gt__(self, other: typing.Union["BaseAgent", ScalarType]) -> bool:
        if isinstance(other, BaseAgent):
            return self.fitness > other.fitness
        elif any(isinstance(other, t) for t in (int, float)):
            return self.fitness > other

        return False

    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __getitem__(self, key: int) -> ScalarType: ...

    @property
    @abc.abstractmethod
    def i(self) -> int: ...

    @property
    @abc.abstractmethod
    def solution(self) -> ArrayType: ...

    @property
    @abc.abstractmethod
    def fitness(self) -> ScalarType: ...

    @property
    @abc.abstractmethod
    def x(self) -> ArrayType: ...

    @abc.abstractmethod
    def __float__(self) -> float: ...

    def __repr__(self) -> str:
        return f"Agent(i={self.i}, fitness={self.fitness}, solution={self.solution})"

    def __hash__(self):
        return hash(self.i)
