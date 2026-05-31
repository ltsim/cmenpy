from cmenpy.agent import MemoryAgent, Agent
from cmenpy.population import Population
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class EpochIteration:
    def __init__(
            self,
            buffer: NDArrayType,
            target: TargetFunction,
            n_it: int,
            n_pop: int,
            r_pop: tuple[int, int] | None = None
    ):
        if r_pop is None:
            r_pop = n_pop, n_pop

        self.__buffer: NDArrayType = buffer
        self.__target: TargetFunction = target
        self.__n_it: int = n_it
        self.__agents: list[Agent] = []
        self.__r_pop: tuple[int, int] = r_pop

        for i in range(n_pop):
            self.__agents.append(MemoryAgent(buffer, i))

        self.__pop: Population = Population(buffer, target, self.__agents, r_pop)

    def __iter__(self):
        for i in range(self.__n_it):
            yield i

    @property
    def population(self) -> Population:
        return self.__pop
