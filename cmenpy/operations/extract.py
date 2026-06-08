import typing

from cmenpy.kernel import KernelBuffer
from cmenpy.operations.stream import ReaderOperator
from cmenpy.types import NDArrayType


class ExtractOperation(ReaderOperator):
    def __init__(self, buffer: KernelBuffer):
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        return self.__buffer.raw_data[item, 1:]

    def __invert__(self) -> NDArrayType:
        return self.__buffer.raw_data[self.__buffer.mask.founds, 1:]

    def __rshift__(self, other: NDArrayType) -> NDArrayType:
        return other

    def __repr__(self) -> str:
        return "Getter<ReaderOperator>()"
