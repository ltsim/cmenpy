import typing

import numpy as np

from cmenpy.kernel.size import KernelSize


class KernelMask:
    """Boolean active/passive mask over the buffer rows.

    A row is *active* while it holds a live individual. The mask starts
    with the first ``size.size`` rows active; deactivating a row (see
    ``PopulationSwarm.remove``) frees it for later insertion.

    Parameters
    ----------
    size : KernelSize
        Supplies the total (max) row count and the initial active count.

    Attributes
    ----------
    active : int
        Number of active rows.
    passive : int
        Number of inactive (free) rows.
    founds : list[bool]
        Copy of the mask as a plain list of booleans.
    """
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
