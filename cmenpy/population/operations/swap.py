from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream import WriterOperator


class SwapOperation(WriterOperator):
    """Row swap writer: ``pop.swap[i] = row`` (moves the old row to the end).

    Parameters
    ----------
    buffer : KernelBuffer
        The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer):
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __setitem__(self, key: int | slice, value: NDArrayType) -> None:
        cp_mem_buff = self.__buffer.raw[key].copy()
        self.__buffer.raw[key] = value
        self.__buffer.raw[:] = cp_mem_buff

    def __repr__(self) -> str:
        return "Swap<WriterOperator>()"
