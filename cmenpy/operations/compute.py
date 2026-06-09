from cmenpy.kernel import KernelBuffer
from cmenpy.operations.stream.vectorizable import (
    VectorizableOperator,
    PipelineOperator,
)
from cmenpy.hints import NDArrayType


class ComputePipeline(PipelineOperator):
    def __init__(self, idx: int, buffer: KernelBuffer):
        self.__idx = idx
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType):
        self.__buffer.apply(value, self.__idx)


class ComputeOperator(VectorizableOperator):
    def __init__(self, buffer: KernelBuffer):
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> "ComputeOperator":
        self.__buffer.apply(value)
        return self

    def __getitem__(self, item: int) -> "PipelineOperator":
        return ComputePipeline(item, self.__buffer)

    def __repr__(self) -> str:
        return "Compute<VectorizableOperator>()"
