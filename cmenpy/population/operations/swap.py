from cmenpy.types import NDArrayType


class SwapOperation:
    def __init__(
        self,
        buffer: NDArrayType,
    ):
        self.__buffer = buffer

    def __setitem__(self, key: int, value: NDArrayType):
        buff_cpy = self.__buffer[key].copy()
        self.__buffer[key] = value

        value[:] = buff_cpy
