from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent.base import BaseAgent
from cmenpy.hints import NDArrayType, ScalarType


class MutableAgent(BaseAgent):
    def __init__(self, n: int, buffer: KernelBuffer):
        self.__i = n
        self.__buffer = buffer

    def __iter__(self):
        return iter(self.__buffer.raw[self.__i, :].copy())

    def __getitem__(self, key: int) -> ScalarType:
        return self.__buffer.raw[self.__i, key + 1]

    @property
    def i(self):
        return self.__i

    @property
    def solution(self):
        return self.__buffer.raw[self.i, 1:]

    @solution.setter
    def solution(self, value: NDArrayType):
        self.__buffer.apply(value, self.__i)

    @property
    def fitness(self):
        return self.__buffer.raw[self.i, 0]

    def __float__(self):
        return self.__buffer.raw[self.i, 0]

    @property
    def x(self) -> NDArrayType:
        return self.__buffer.raw[self.i, :]
