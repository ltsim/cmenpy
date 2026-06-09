import abc

from cmenpy.types import NDArrayType


class BaseAgent(abc.ABC):
    def __lt__(self, other):
        if isinstance(other, BaseAgent):
            return self.fitness < other.fitness
        elif any(isinstance(other, t) for t in (int, float)):
            return self.fitness < other

        return False

    def __gt__(self, other):
        if isinstance(other, BaseAgent):
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

    @property
    @abc.abstractmethod
    def x(self) -> NDArrayType: ...

    @abc.abstractmethod
    def __float__(self) -> float: ...

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, fitness={self.fitness}, solution={self.solution})"

    def __hash__(self):
        return hash(self.id)
