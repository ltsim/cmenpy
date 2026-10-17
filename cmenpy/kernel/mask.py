"""Boolean active/passive mask over the kernel buffer rows."""

import typing

import numpy as np

from cmenpy.kernel.size import KernelSize


class KernelMask:
    """Boolean active/passive mask over the buffer rows.

    A row is *active* while it holds a live individual. The mask starts
    with the first ``size.size`` rows active; deactivating a row (see
    ``PopulationSwarm.remove``) frees it for later insertion.

    Args:
        size: Supplies the total (max) row count and the initial active count.

    Attributes:
        active: Number of active rows.
        passive: Number of inactive (free) rows.
        founds: Copy of the mask as a plain list of booleans.
    """

    def __init__(
        self,
        size: KernelSize,
    ):
        """Initialize the mask with the first ``size.size`` rows active."""
        self.__size: KernelSize = size
        self.__mask = np.fromiter((False for _ in range(size.max)), dtype=bool)
        self.__mask[0 : self.__size.size] = True

    def __iter__(self) -> typing.Iterator[bool]:
        """Iterate over the mask as plain booleans."""
        return iter(map(bool, self.__mask))

    def __setitem__(
        self,
        key: int | slice | list[int] | None,
        value: bool,
    ) -> None:
        """Set the activity flag of the rows selected by ``key``."""
        self.__mask[key] = value

    def __getitem__(self, key: int | slice | list[int] | None) -> typing.Any:
        """Return the activity flag of the rows selected by ``key``."""
        return self.__mask[key]

    @property
    def active(self) -> int:
        """Number of active rows."""
        return sum(self.__mask)

    @property
    def passive(self) -> int:
        """Number of inactive (free) rows."""
        return len(self.__mask) - sum(self.__mask)

    @property
    def founds(self) -> list[bool]:
        """Copy of the mask as a plain list of booleans."""
        return [*map(bool, self.__mask)]
