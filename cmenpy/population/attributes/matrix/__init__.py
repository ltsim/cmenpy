import abc

import numpy.typing as npt

from cmenpy.hints import NDArrayType
from cmenpy.population.attributes.matrix.mixin import (
    BufferMatrixAccess,
    BufferMatrixCompute,
)


class BufferMatrixAbstract(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, item: int) -> NDArrayType: ...

    @abc.abstractmethod
    def __array__(
        self, dtype: "npt.DTypeLike | None" = None, copy: bool | None = None
    ) -> NDArrayType: ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def __str__(self) -> str: ...


__all__ = ["BufferMatrixAbstract", "BufferMatrixAccess", "BufferMatrixCompute"]
