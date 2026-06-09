import abc

from cmenpy.hints import NDArrayType, DType


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
    def i(self) -> int: ...

    @property
    @abc.abstractmethod
    def solution(self) -> NDArrayType: ...

    @property
    @abc.abstractmethod
    def fitness(self) -> DType: ...

    @property
    @abc.abstractmethod
    def x(self) -> NDArrayType: ...

    @abc.abstractmethod
    def __float__(self) -> float: ...

    def __repr__(self) -> str:
        return f"Agent(i={self.i}, fitness={self.fitness}, solution={self.solution})"

    def __hash__(self):
        return hash(self.i)
