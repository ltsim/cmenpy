from cmenpy.agent.base import BaseAgent
from cmenpy.target import TargetFunction
from cmenpy.hints import NDArrayType


class MutableAgent(BaseAgent):
    def __init__(self, buffer: NDArrayType, n: int, target: TargetFunction):
        self.__buffer = buffer
        self.__i = n
        self.__target = target

    def __iter__(self):
        return iter(self.__buffer.copy())

    def __getitem__(self, key: int) -> float:
        return self.__buffer[self.__i, key]

    @property
    def i(self):
        return self.__i

    @property
    def solution(self):
        return self.__buffer[self.i, 1:]

    @solution.setter
    def solution(self, value: NDArrayType):
        self.__buffer[self.i] = self.__target.evaluate(value)

    @property
    def fitness(self):
        return self.__buffer[self.i, 0]

    def __float__(self):
        return self.__buffer[self.i, 0]

    @property
    def x(self) -> NDArrayType:
        return self.__buffer[self.i, :]
