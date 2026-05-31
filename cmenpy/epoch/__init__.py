from cmenpy.agent import MemoryAgent, Agent
from cmenpy.population import Population
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class EpochIteration:
    def __init__(
            self,
            buffer: NDArrayType,
            target: TargetFunction,
            epochs: int,
            pop_size: int,
            pop_range: tuple[int, int] | None = None
    ):
        if pop_range is None:
            pop_range = pop_size, pop_size

        self.__buffer: NDArrayType = buffer
        self.__target: TargetFunction = target
        self.__epochs: int = epochs
        self.__agents: list[Agent] = []
        self.__pop_range: tuple[int, int] = pop_range
        self.__current_epoch: int = 0

        for i in range(pop_size):
            self.__agents.append(MemoryAgent(buffer, i))

        self.__pop: Population = Population(buffer, target, self.__agents, pop_range)

    def __iter__(self):
        for i in range(0, self.__epochs):
            self.__current_epoch = i
            yield i

    @property
    def population(self) -> Population:
        return self.__pop

    @property
    def current(self):
        return self.__current_epoch

    @property
    def max_epochs(self):
        return self.__epochs
