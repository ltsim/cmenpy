import abc
import functools
import typing

import numpy as np

from cmenpy import low


class Bounds:
    def __init__(self, bounds):
        self.__bounds = bounds

    @functools.cached_property
    def ndim(self):
        return len(self.__bounds)

    def __iter__(self):
        return (b for b in self.__bounds)


class TargetFunction:
    def __init__(self, f, bounds: Bounds):
        self.__f = f
        self.__n_call = 0
        self.__bounds = bounds

    def __call__(self, x, *args, **kwargs):
        self.__n_call += 1

        return self.__f(x)

    def wrapper(self):
        def inner(x):
            self.__n_call += 1

            return self.__f(x)

        return inner

    @property
    def n_call(self):
        return self.__n_call

    @property
    def bounds(self):
        return self.__bounds


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

    @property
    @abc.abstractmethod
    def id(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def solution(self):
        ...

    @property
    @abc.abstractmethod
    def fitness(self) -> int | float:
        ...

    @abc.abstractmethod
    def __float__(self) -> float:
        ...

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, fitness={self.fitness}, solution={self.solution})"

    def __hash__(self):
        return hash(self.id)


class MemoryAgent(Agent):
    def __init__(self, buffer: np.ndarray, n: int):
        self.__buffer = buffer
        self.__id = n

    @property
    def id(self):
        return self.__id

    @property
    def solution(self):
        return self.__buffer[self.id, 1:]

    @solution.setter
    def solution(self, value):
        self.__buffer[self.id, 1:] = value

    @property
    def fitness(self):
        return self.__buffer[self.id, 0]

    def __float__(self):
        return self.__buffer[self.id, 0]


class VirtualAgent(Agent):
    def __init__(self, buffer: np.ndarray, n: int):
        self.__buffer = buffer[n, :].copy()
        self.__id = n

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


C = typing.TypeVar("C", bound=Agent)


class Population(typing.Generic[C]):
    def __init__(self, buffer: np.ndarray, target: TargetFunction, agents: list[Agent], r_pop: tuple[int, int],
                 d_class: typing.Type[C] = VirtualAgent):
        n_pop = len(agents)

        self.__min_pop, self.__max_pop = r_pop
        self.__buffer = buffer
        self.__target = target
        self.__agents = agents
        self.__mask = (
                list(True for _ in range(0, n_pop)) + list(False for _ in range(n_pop, self.__max_pop))
        )

        self.__d_class = d_class

    def __len__(self):
        return len(self.__agents)

    def __iter__(self):
        return (a for a in self.__agents)

    def __getitem__(self, i: int) -> VirtualAgent:
        a = self.__agents[i]

        return VirtualAgent(
            self.__buffer,
            a.id
        )

    def __setitem__(self, i, value):
        if isinstance(value, Agent):
            self.__agents[i] = value
        elif any(isinstance(value, n) for n in (list, tuple, np.ndarray)):
            self.__buffer[i, :] = value

    def remove(self, i: int):
        founds = [*filter(lambda a: a.id == i, self.__agents)]

        if not len(founds) > 0:
            raise ValueError("Agent not found.")

        agent = founds[0]

        self.__mask[i] = False
        self.__buffer[i, :] = np.nan
        self.__agents.remove(agent)

    def append(self, solution=None):
        founds = [i for i, x in enumerate(self.__mask) if not x]

        if not len(founds) > 0:
            raise ValueError("Max agents in memory.")

        i = founds[0]

        self.__mask[i] = True
        self.__agents.append(
            MemoryAgent(self.__buffer, i)
        )
        self.__buffer[i, 1:] = solution
        self.__buffer[i, 0] = np.apply_along_axis(self.__target, 0, solution)

        return VirtualAgent(
            self.__buffer,
            i
        )

    def __invert__(self):
        return self.__buffer[self.__mask, 1:].copy()

    def __matmul__(self, other):
        buff = self.__buffer.copy()
        buff[self.__mask, 1:] = other
        buff[self.__mask, 0] = np.apply_along_axis(self.__target, 1, self.__buffer[self.__mask, 2:])

        return buff[self.__mask, 2:]

    def __imatmul__(self, other):
        self.__buffer[self.__mask, 1:] = other
        self.__buffer[self.__mask, 0] = np.apply_along_axis(self.__target, 1, self.__buffer[self.__mask, 2:])

        return self

    @property
    def best(self):
        b_pop = sorted(self.__agents, key=lambda a: a.fitness)[0]

        return VirtualAgent(
            self.__buffer,
            b_pop.id
        )

    @property
    def worst(self):
        w_pop = sorted(self.__agents, key=lambda a: a.fitness)[-1]

        return VirtualAgent(
            self.__buffer,
            w_pop.id
        )

    @property
    def free_space(self):
        return len(self.__agents) < self.max_pop

    @property
    def min_pop(self):
        return self.__min_pop

    @property
    def max_pop(self):
        return self.__max_pop


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


class AlgorithmModel:
    def __init__(self, alias: str, seed=None):
        self.__alias = alias
        self.__inner = None
        self.__buffer = None
        self.__epoch = None

    def __del__(self):
        pass

    def define(self, func):
        @functools.wraps(func)
        def inner(f, bounds, n_it, n_pop, r_pop=None):
            if r_pop is None:
                r_pop = (n_pop, n_pop)

            min_pop, max_pop = r_pop

            if min_pop > max_pop:
                raise IndexError("Population size is too small.")
            elif n_pop < min_pop:
                raise IndexError("Population size is too small.")
            elif n_pop > max_pop:
                raise IndexError("Population size is too large.")

            bounds = Bounds(bounds)
            target = TargetFunction(f, bounds)

            self.__buffer = low.init_buffer(max_pop, bounds.ndim)
            self.__epoch = EpochIteration(self.__buffer, target, n_it, n_pop, r_pop)

            return func(self.__epoch.population, bounds, self.__epoch)

        self.__inner = inner

        return inner

    def __call__(self, *args, **kwargs):
        return self.__inner(*args, **kwargs)


def define(alias: str):
    model = AlgorithmModel(alias)

    def inner(func):
        return model.define(func)

    return inner
