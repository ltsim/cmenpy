import typing

import numpy as np

from cmenpy.kernel.size import KernelSize


class KernelMask:
    def __init__(
        self,
        size: KernelSize,
    ):
        self.__size: KernelSize = size
        self.__mask = np.fromiter((False for _ in range(size.max)), dtype=bool)
        self.__mask[0 : self.__size.size] = True

    def __iter__(self) -> typing.Iterator[bool]:
        return iter(map(bool, self.__mask))

    def __setitem__(
        self,
        key: int | slice | list[int] | None,
        value: bool,
    ) -> None:
        self.__mask[key] = value

    def __getitem__(
        self, key: int | slice | list[int] | None
    ) -> typing.Any:
        return self.__mask[key]

    @property
    def active(self) -> int:
        return sum(self.__mask)

    @property
    def passive(self) -> int:
        return len(self.__mask) - sum(self.__mask)

    @property
    def founds(self) -> list[bool]:
        return [*map(bool, self.__mask)]
