from cmenpy.agent.base import BaseAgent
from cmenpy.types import NDArrayType


class ImmutableAgent(BaseAgent):
    def __init__(self, buffer: NDArrayType, n: int):
        self.__buffer = buffer[n, :].copy()
        self.__id = n

    def __iter__(self):
        return iter(self.__buffer.copy())

    def __getitem__(self, key: int) -> float:
        return self.__buffer[key]

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

    @property
    def x(self) -> NDArrayType:
        return self.__buffer
