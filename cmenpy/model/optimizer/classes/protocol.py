import typing

from cmenpy.bounds import Bounds
from cmenpy.population import Population


@typing.runtime_checkable
class ModelProtocol(typing.Protocol):
    epoch: int
    population: int
    nfe: int

    def initialize(self, population: Population, bounds: Bounds):
        ...

    def evolve(self, e: int, population: Population, bounds: Bounds) -> None:
        ...
