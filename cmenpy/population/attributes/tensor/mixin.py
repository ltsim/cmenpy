import abc

from cmenpy.hints import NDArrayType
import typing


class TensorAccess(abc.ABC):
    @abc.abstractmethod
    def __invert__(self) -> NDArrayType: ...


class TensorCompute(abc.ABC):
    @abc.abstractmethod
    def __lshift__(self, other: NDArrayType) -> typing.Self: ...
