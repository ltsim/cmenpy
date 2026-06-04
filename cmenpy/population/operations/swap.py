from cmenpy.population.operations.stream import WriterOperator
from cmenpy.types import NDArrayType


class SwapOperation(WriterOperator):
    def __init__(
        self,
        buffer: NDArrayType,
    ):
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __setitem__(self, key: int, value: NDArrayType):
        buff_cpy = self.__buffer[key].copy()
        self.__buffer[key] = value

        value[:] = buff_cpy

    def __repr__(self) -> str:
        return "Swap<WriterOperator>()"
