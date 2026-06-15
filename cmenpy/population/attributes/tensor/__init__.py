import abc

from cmenpy.hints import ArrayType
from cmenpy.population.attributes.tensor.mixin import TensorCompute, TensorAccess


class TensorAbstract(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, item: int) -> ArrayType: ...

    @abc.abstractmethod
    def __array__(self, dtype=None, copy=None) -> ArrayType: ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def __str__(self) -> str: ...


__all__ = ["TensorAbstract", "TensorCompute", "TensorAccess"]
