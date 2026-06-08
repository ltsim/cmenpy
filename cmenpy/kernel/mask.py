import typing

import numpy as np

from cmenpy.kernel.size import KernelSize


class KernelMask:
    def __init__(
        self,
        size: KernelSize,
    ):
        self.__size: KernelSize = size
        self.__mask = np.fromiter((False for _ in range(0, size.max)), dtype=bool)
        self.__mask[0 : self.__size.size] = True

    def __iter__(self):
        return iter(map(bool, self.__mask))

    def __setitem__(
        self,
        key: typing.Union[int, slice, typing.List[int], None],
        value: bool,
    ):
        self.__mask[key] = value

    def __getitem__(
        self, key: typing.Union[int, slice, typing.List[int], None]
    ) -> typing.Any:
        return self.__mask[key]

    @property
    def active(self):
        return sum(self.__mask)

    @property
    def passive(self):
        return len(self.__mask) - sum(self.__mask)

    @property
    def founds(self) -> list[bool]:
        return [*map(bool, self.__mask)]
