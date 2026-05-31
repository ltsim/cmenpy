import typing

from cmenpy.bounds import Bounds
from cmenpy.population import Population


@typing.runtime_checkable
class ModelProtocol(typing.Protocol):
    def initialize(self, population: Population, bounds: Bounds) -> None: ...

    def evolve(self, e: int, population: Population, bounds: Bounds) -> None: ...
