from cmenpy.kernel import KernelBuffer
from cmenpy.tensor.operations.stream import ReaderOperator
from cmenpy.hints import NDArrayType


class ExtractOperation(ReaderOperator):
    def __init__(self, buffer: KernelBuffer):
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        return self.__buffer.raw[item, 1:]

    def __invert__(self) -> NDArrayType:
        return self.__buffer.raw[self.__buffer.mask.founds, 1:]

    def __rshift__(self, other: NDArrayType) -> NDArrayType:
        return other

    def __repr__(self) -> str:
        return "Extract<ReaderOperator>()"
