from cmenpy.population.operations.stream.vectorizable import (
    VectorizableOperator,
    PipelineOperator,
)
from cmenpy.types import NDArrayType


class AssignPipeline(PipelineOperator):
    def __init__(self, idx: int, buffer: NDArrayType):
        self.__idx = idx
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType):
        self.__buffer[self.__idx] = value


class AssignOperator(VectorizableOperator):
    def __init__(
        self,
        mask: list[int],
        buffer: NDArrayType,
    ):
        super().__init__(buffer.shape)
        self.__mask = mask
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> "VectorizableOperator":
        if self.shape != value.shape:
            raise IndexError(
                "The buffer cannot be assigned a value; the sizes do not match."
            )

        self.__buffer[:] = value

        return self

    def __getitem__(self, item: int) -> "PipelineOperator":
        return AssignPipeline(item, self.__buffer)

    def __repr__(self) -> str:
        return "Assign<VectorizableOperator>()"
