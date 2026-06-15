from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream import ReaderOperator
from cmenpy.hints import ArrayType


class AccessOperator(ReaderOperator):
    def __init__(
        self,
        buffer: KernelBuffer,
    ):
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> ArrayType:
        return self.__buffer.raw[item, :]

    def __invert__(self) -> ArrayType:
        return self.__buffer.raw[self.__buffer.mask.founds, :]

    def __rshift__(self, other: ArrayType) -> ArrayType:
        return other

    def __repr__(self) -> str:
        return "Access<ReaderOperator>()"
