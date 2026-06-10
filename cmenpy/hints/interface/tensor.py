import abc

from cmenpy.hints import NDArrayType


class TensorABC(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, item: int) -> NDArrayType: ...

    @abc.abstractmethod
    def __array__(self, dtype=None, copy=None) -> NDArrayType: ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def __str__(self) -> str: ...
