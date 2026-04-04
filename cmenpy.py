import abc
import functools
import collections

import numpy as np


def expand_buffer(n: int, ndim: int) -> np.ndarray:
    return np.hstack(([n], np.full((ndim + 1), np.nan)))


def init_buffer(n_pop: int, ndim: int) -> np.ndarray:
    return np.full((n_pop, ndim + 2), np.nan)


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


class MemoryAgent(Agent):
    def __init__(self, buffer: np.ndarray, n: int):
        self.__buffer = buffer
        self.__id = n

    @property
    def id(self):
        return self.__id

    @property
    def solution(self):
        return self.__buffer[self.mask, 2:]

    @solution.setter
    def solution(self, value):
        self.__buffer[self.mask, 2:] = value

    @property
    def fitness(self):
        return self.__buffer[self.mask, 1]

    @property
    def mask(self):
        return self.__buffer[:, 0] == self.id

    def __float__(self):
        return self.__buffer[self.mask, 1]


class VirtualAgent(Agent):
    def __init__(self, buffer: np.ndarray, n: int):
        mask = buffer[:, 0] == n

        self.__buffer = buffer[mask, :].view()
        self.__id = n

    @property
    def id(self):
        return self.__id

    @property
    def solution(self):
        return self.__buffer[2:]

    @property
    def fitness(self):
        return self.__buffer[1]

    def __float__(self):
        pass


class Population:
    def __init__(self, buffer: np.ndarray, target: TargetFunction, agents: list[Agent], r_pop=None):
        self.__a_buffer = collections.deque(n for n in range(*r_pop))
        self.__buffer = buffer
        self.__target = target
        self.__agents = agents

    def __len__(self):
        return len(self.__agents)

    def __iter__(self):
        return (a for a in self.__agents)

    def __getitem__(self, i):
        return self.__agents[i]

    def __setitem__(self, i, value):
        self.__agents[i] = value

    def remove(self, i: int):
        agents = [*filter(lambda a: a.id == i, self.__agents)]

        if not len(agents) > 0:
            raise IndexError("Agent not found.")

        agent = agents[0]

        n_buff = np.delete(self.__buffer, i, axis=0)
        self.__buffer.resize(n_buff.shape, refcheck=False)
        self.__buffer[:] = n_buff

        self.__agents.remove(agent)

    def append(self, solution=None):
        n_current_pop = int(np.argmax(self.__buffer[:, 0]) + 1)
        n_buff_pop = expand_buffer(n_current_pop, self.__target.bounds.ndim)

        if solution is not None:
            n_buff_pop[2:] = solution
            n_buff_pop[1] = self.__target(solution)

        n_buff = np.vstack((self.__buffer, n_buff_pop))

        self.__buffer.resize(n_buff.shape, refcheck=False)
        self.__buffer[:] = n_buff

        agent = MemoryAgent(self.__buffer, n_current_pop)

        self.__agents.append(agent)

        return VirtualAgent(
            self.__buffer,
            n_current_pop
        )

    def __invert__(self):
        return self.__buffer[:, 2:].copy()

    def __matmul__(self, other):
        buff = self.__buffer.copy()
        buff[:, 2:] = other
        buff[:, 1] = np.apply_along_axis(self.__target, 1, self.__buffer[:, 2:])

        return buff[:, 2:]

    def __imatmul__(self, other):
        self.__buffer[:, 2:] = other
        self.__buffer[:, 1] = np.apply_along_axis(self.__target, 1, self.__buffer[:, 2:])
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


class EpochIteration:
    def __init__(self, buffer, target, n_it: int, n_pop: int, r_pop=None):
        if r_pop is None:
            r_pop = (n_pop, n_pop)

        self.__buffer = buffer
        self.__target = target
        self.__n_it = n_it
        self.__agents = []
        self.__pop = Population(buffer, target, self.__agents)
        self.__r_pop = r_pop

        for i in range(n_pop):
            buffer[i, :1] = i
            self.__agents.append(MemoryAgent(buffer, i))

    def __iter__(self):
        for i in range(self.__n_it):
            yield i

            """
            n_agents = []

            for a in self.__agents:
                n_agents.append(Agent(self.__buffer, i, a.id))

            self.__agents = n_agents
            self.__pop = Population(self.__buffer, self.__target, i, self.__agents)
            """

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
        def inner(f, bounds, n_it, n_pop):
            bounds = Bounds(bounds)
            target = TargetFunction(f, bounds)

            self.__buffer = init_buffer(n_pop, bounds.ndim)
            self.__epoch = EpochIteration(self.__buffer, target, n_it, n_pop)

            return func(self.__epoch.population, bounds, self.__epoch)

        self.__inner = inner

        return inner


def define(alias: str):
    model = AlgorithmModel(alias)

    def inner(func):
        return model.define(func)

    return inner
