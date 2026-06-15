import abc

from cmenpy.hints import ArrayType
import typing


class TensorAccess(abc.ABC):
    @abc.abstractmethod
    def __invert__(self) -> ArrayType: ...


class TensorCompute(abc.ABC):
    @abc.abstractmethod
    def __lshift__(self, other: ArrayType) -> typing.Self: ...
