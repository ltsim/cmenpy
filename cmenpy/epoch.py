from cmenpy.agent import MemoryAgent
from cmenpy.population import Population


class EpochIteration:
    def __init__(self, buffer, target, n_it: int, n_pop: int, r_pop):
        if r_pop is None:
            r_pop = (n_pop, n_pop)

        self.__buffer = buffer
        self.__target = target
        self.__n_it = n_it
        self.__agents = []
        self.__r_pop = r_pop

        for i in range(n_pop):
            self.__agents.append(MemoryAgent(buffer, i))

        self.__pop = Population(buffer, target, self.__agents, r_pop)

    def __iter__(self):
        for i in range(self.__n_it):
            yield i

    @property
    def population(self):
        return self.__pop
