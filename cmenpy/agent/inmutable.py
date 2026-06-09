from cmenpy.kernel import KernelBuffer
from cmenpy.agent.base import BaseAgent
from cmenpy.hints import NDArrayType, ScalarType


class ImmutableAgent(BaseAgent):
    def __init__(self, n: int, buffer: KernelBuffer):
        self.__i = n
        self.__buffer = buffer.raw[n, :].copy()

    def __iter__(self):
        return iter(self.__buffer.copy())

    def __getitem__(self, key: int) -> ScalarType:
        return self.__buffer[key + 1]

    @property
    def i(self):
        return self.__i

    @property
    def solution(self):
        return self.__buffer[1:]

    @property
    def fitness(self):
        return self.__buffer[0]

    def __float__(self):
        return self.__buffer[0]

    @property
    def x(self) -> NDArrayType:
        return self.__buffer
