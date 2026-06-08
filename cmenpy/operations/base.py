import abc

from cmenpy.operations import (
    SwapOperation,
    ComputeOperator,
    AssignOperator,
    ExtractOperation,
)
from cmenpy.operations.access import AccessOperator


class OperationBase(abc.ABC):
    @property
    @abc.abstractmethod
    def swap(self) -> SwapOperation: ...

    @property
    @abc.abstractmethod
    def compute(self) -> ComputeOperator: ...

    @property
    @abc.abstractmethod
    def assign(self) -> AssignOperator: ...

    @property
    @abc.abstractmethod
    def extract(self) -> ExtractOperation: ...

    @property
    @abc.abstractmethod
    def access(self) -> AccessOperator: ...
