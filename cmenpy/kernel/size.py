import typing


class KernelSize:
    def __init__(
        self,
        size: int,
        min_size: typing.Optional[int] = None,
        max_size: typing.Optional[int] = None,
    ):
        if min_size is None:
            min_size = size

        if max_size is None:
            max_size = size

        if size < min_size or size > max_size:
            raise ValueError("Size must be between 'min_size' and 'max_size'.")

        self.__size = size
        self.__min = min_size
        self.__max = max_size

    @property
    def max(self) -> int:
        return self.__max

    @property
    def min(self) -> int:
        return self.__min

    @property
    def size(self) -> int:
        return self.__size
