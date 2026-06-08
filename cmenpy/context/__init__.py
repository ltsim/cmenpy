import typing

import numpy as np

from cmenpy.kernel import low, KernelBuffer, KernelSize
from cmenpy.bounds import Bounds
from cmenpy.epoch import EpochIteration
from cmenpy.generator import DefaultGenerator
from cmenpy.population import PopulationSwarm
from cmenpy.target import Target, TargetFunction
from cmenpy.tracker import Tracker
from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType


class Context:
    def __init__(
        self,
        f: Target,
        bounds: Bounds,
        buffer: KernelBuffer,
        generator: DefaultGenerator,
        sense: SenseType,
    ):
        self.__target = TargetFunction(f, bounds)
        self.__bounds = bounds
        self.__buff = buffer
        self.__generator = generator
        self.__sense = sense

    @property
    def target(self) -> TargetFunction:
        return self.__target

    @property
    def bounds(self) -> Bounds:
        return self.__bounds

    @property
    def buff(self) -> KernelBuffer:
        return self.__buff

    @property
    def rng(self) -> np.random.Generator:
        return self.__generator.rng

    @property
    def sense(self) -> SenseType:
        return self.__sense


class MainContextManager:
    def __init__(
        self,
        f: Target,
        bounds: Bounds,
        epochs: int,
        size: KernelSize,
        seed: typing.Optional[int] = None,
        debug: bool = False,
        sense: SenseType = "min",
    ):
        self.__size = size
        self.__target = TargetFunction(f, bounds)
        self.__bounds = bounds
        self.__sense = sense
        self.__k_buff = KernelBuffer(size, self.__bounds, self.__target, self.__sense)
        self.__generator = DefaultGenerator(seed=seed)
        self.__tracker = Tracker(self.__k_buff)
        self.__epoch_it = EpochIteration(epochs, self.__tracker, debug)

    @property
    def sense(self):
        return self.__sense

    @property
    def epoch_it(self) -> EpochIteration:
        return self.__epoch_it

    @property
    def buffer(self) -> KernelBuffer:
        return self.__k_buff

    @property
    def target(self) -> TargetFunction:
        return self.__target

    @property
    def bounds(self) -> Bounds:
        return self.__bounds

    @property
    def default_generator(self) -> DefaultGenerator:
        return self.__generator

    @property
    def tracker(self) -> Tracker:
        return self.__tracker
