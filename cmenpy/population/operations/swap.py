from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream import WriterOperator
from cmenpy.hints import NDArrayType


class SwapOperation(WriterOperator):
    def __init__(self, buffer: KernelBuffer):
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __setitem__(self, key: int, value: NDArrayType):
        cp_mem_buff = self.__buffer.raw[key].copy()
        self.__buffer.raw[key] = value
        self.__buffer.raw[:] = cp_mem_buff

    def __repr__(self) -> str:
        return "Swap<WriterOperator>()"
