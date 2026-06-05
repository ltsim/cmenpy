import abc

from cmenpy.population.operations.extract import ExtractOperation
from cmenpy.population.operations.assign import AssignOperator
from cmenpy.population.operations.compute import ComputeOperator
from cmenpy.population.operations.swap import SwapOperation


class PopulationOperationBase(abc.ABC):
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
