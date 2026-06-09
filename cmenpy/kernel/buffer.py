import typing

import numpy as np

from cmenpy.bounds import Bounds
from cmenpy.kernel.index import BufferIndex
from cmenpy.kernel.low import init_buffer
from cmenpy.kernel.mask import KernelMask
from cmenpy.kernel.size import KernelSize
from cmenpy.target import TargetFunction
from cmenpy.hints import NDArrayType
from cmenpy.hints.option import SenseType


class KernelBuffer:
    def __init__(
        self,
        size: KernelSize,
        bounds: Bounds,
        target: TargetFunction,
        sense: SenseType,
    ):
        self.__size = size
        self.__bounds = bounds
        self.__target = target
        self.__sense = sense

        self.__buffer: NDArrayType = init_buffer(self.__size.max, bounds.ndim)
        self.__mask = KernelMask(self.__size)

    def apply(
        self,
        x: NDArrayType,
        idx: typing.Union[int, slice, typing.List[int], None] = None,
    ) -> None:
        """Vectorizes the target function to evaluate and update the solutions matrix.

        The internal buffer stores "fitness" in column 0 and "solutions" or "x"
        from column 1 onwards. Supports updating a single row, a slice, a list/array
        of indices, or the entire active matrix if `idx` is None.

        Parameters
        ----------
        x : NDArrayType
            An array containing the new solution data to be inserted into the buffer.
        idx : int, slice, NDArrayType, list, optional
            The row index, slice, or index mask to update. If None, defaults to
            the internal mask.

        Returns
        -------
        None
        """
        rows = self.__mask.founds if idx is None else idx
        self.__buffer[rows, 1:] = x

        if isinstance(idx, int):
            self.__buffer[rows, 0] = self.__target.evaluate(x)
        else:
            self.__buffer[rows, 0] = np.apply_along_axis(
                self.__target, 1, self.__buffer[rows, 1:]
            )

    @property
    def shape(self):
        return self.__buffer.shape

    @property
    def mask(self) -> KernelMask:
        return self.__mask

    @property
    def raw(self) -> NDArrayType:
        return self.__buffer

    @property
    def idx(self) -> BufferIndex:
        return BufferIndex(self.__buffer, self.__mask, self.__sense)

    @property
    def size(self) -> KernelSize:
        return self.__size

    @property
    def snapshot(self) -> "KernelBuffer":
        return KernelBuffer(self.__size, self.__bounds, self.__target, self.__sense)
