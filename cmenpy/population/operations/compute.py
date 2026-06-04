import numpy as np

from cmenpy.population.operations.stream import VectorizableOperator
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class ComputeOperator(VectorizableOperator):
    def __init__(self, mask: list[int], buffer: NDArrayType, target: TargetFunction):
        super().__init__(buffer.shape)
        self.__mask = mask
        self.__buffer = buffer
        self.__target = target

    def __imatmul__(self, other: NDArrayType):
        self.__buffer[self.__mask, 1:] = other
        self.__buffer[self.__mask, 0] = np.apply_along_axis(
            self.__target, 1, self.__buffer[self.__mask, 1:]
        )

        return self

    def __matmul__(self, other):
        buff = self.__buffer.copy()
        buff[self.__mask, 1:] = other
        buff[self.__mask, 0] = np.apply_along_axis(
            self.__target, 1, self.__buffer[self.__mask, 1:]
        )

        return buff[self.__mask, 1:]

    def __setitem__(self, item, value):
        self.__buffer[item, 1:] = value
        self.__buffer[item, 0] = self.__target.evaluate(value)

    def __repr__(self) -> str:
        return "Compute<VectorizableOperator>()"
