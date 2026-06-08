from cmenpy.kernel import KernelBuffer
from cmenpy.operations.stream import WriterOperator
from cmenpy.types import NDArrayType


class SwapOperation(WriterOperator):
    def __init__(self, buffer: KernelBuffer):
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __setitem__(self, key: int, value: NDArrayType):
        cp_mem_buff = self.__buffer.buffer[key].copy()
        self.__buffer.buffer[key] = value
        self.__buffer.buffer[:] = cp_mem_buff

    def __repr__(self) -> str:
        return "Swap<WriterOperator>()"
