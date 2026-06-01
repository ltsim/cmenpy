import typing

import numpy as np

from cmenpy import low
from cmenpy.agent import MemoryAgent, Agent
from cmenpy.bounds import Bounds
from cmenpy.epoch import EpochIteration
from cmenpy.generator import DefaultGenerator
from cmenpy.population import Population
from cmenpy.target import Target, TargetFunction
from cmenpy.types import NDArrayType


class Context:
    def __init__(
        self,
        f: Target,
        bounds: Bounds,
        population: Population,
        generator: DefaultGenerator,
    ):
        self.__target = TargetFunction(f, bounds)
        self.__bounds = bounds
        self.__population = population
        self.__generator = generator

    @property
    def target(self) -> TargetFunction:
        return self.__target

    @property
    def bounds(self) -> Bounds:
        return self.__bounds

    @property
    def pop(self) -> Population:
        return self.__population

    @property
    def rng(self) -> np.random.Generator:
        return self.__generator.rng


class MainContextManager:
    def __init__(
        self,
        f: Target,
        bounds: Bounds,
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
        seed: typing.Optional[int] = None,
    ):
        if pop_range is None:
            pop_range = pop_size, pop_size

        min_pop, max_pop = pop_range

        self.__agents: list[Agent] = []
        self.__target = TargetFunction(f, bounds)
        self.__buffer = low.init_buffer(max_pop, bounds.ndim)
        self.__epoch_it = EpochIteration(epochs)
        self.__bounds = bounds
        self.__generator = DefaultGenerator(seed=seed)

        for i in range(pop_size):
            self.__agents.append(MemoryAgent(self.__buffer, i))

        self.__population: Population = Population(
            self.__buffer, self.__target, self.__agents, pop_range
        )

    @property
    def agents(self) -> typing.List[Agent]:
        return self.__agents

    @property
    def population(self) -> Population:
        return self.__population

    @property
    def epoch_it(self) -> EpochIteration:
        return self.__epoch_it

    @property
    def buffer(self) -> NDArrayType:
        return self.__buffer

    @property
    def target(self) -> TargetFunction:
        return self.__target

    @property
    def bounds(self) -> Bounds:
        return self.__bounds

    @property
    def default_generator(self) -> DefaultGenerator:
        return self.__generator
