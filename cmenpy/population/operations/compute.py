import numpy as np

from cmenpy.population.operations.stream.vectorizable import (
    VectorizableOperator,
    PipelineOperator,
)
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class ComputePipeline(PipelineOperator):
    def __init__(self, idx: int, buffer: NDArrayType, target: TargetFunction):
        self.__idx = idx
        self.__buffer = buffer
        self.__target = target

    def __lshift__(self, value: NDArrayType):
        self.__buffer[self.__idx, 1:] = value
        self.__buffer[self.__idx, 0] = self.__target.evaluate(value)


class ComputeOperator(VectorizableOperator):
    def __init__(self, mask: list[int], buffer: NDArrayType, target: TargetFunction):
        super().__init__(buffer.shape)
        self.__mask = mask
        self.__buffer = buffer
        self.__target = target

    def __lshift__(self, value: NDArrayType):
        self.__buffer[self.__mask, 1:] = value
        self.__buffer[self.__mask, 0] = np.apply_along_axis(
            self.__target, 1, self.__buffer[self.__mask, 1:]
        )

        return self

    def __getitem__(self, item: int) -> "PipelineOperator":
        return ComputePipeline(item, self.__buffer, self.__target)

    def __repr__(self) -> str:
        return "Compute<VectorizableOperator>()"
