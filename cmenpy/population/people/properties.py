import numpy as np

from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent import MutableAgent
from cmenpy.types.option import SenseType


class GlobalPopulationProperty:
    def __init__(
        self,
        buffer: KernelBuffer,
    ):
        self.__buffer = buffer

    @property
    def fitnesses(self): ...

    @property
    def solutions(self): ...

    @property
    def best(self) -> MutableAgent: ...

    @property
    def worst(self) -> MutableAgent: ...
