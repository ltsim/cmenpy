import typing

from cmenpy.population.operations.stream import ReaderOperator
from cmenpy.types import NDArrayType


class ExtractOperation(ReaderOperator):
    def __init__(
        self,
        mask: typing.List[bool],
        buffer: NDArrayType,
    ):
        super().__init__(buffer.shape)
        self.__mask = mask
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        return self.__buffer[item, 1:]

    def __invert__(self) -> NDArrayType:
        return self.__buffer[self.__mask, 1:]

    def __rshift__(self, other: NDArrayType) -> NDArrayType:
        return other

    def __repr__(self) -> str:
        return "Getter<ReaderOperator>()"
