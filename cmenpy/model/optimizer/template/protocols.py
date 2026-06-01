import typing

from cmenpy.bounds import Bounds
from cmenpy.epoch import Epoch
from cmenpy.population import Population
from cmenpy.context import MainContextManager, Context


@typing.runtime_checkable
class ModelProtocol(typing.Protocol):
    def initialize(self, ctx: Context) -> None: ...

    def evolve(self, e: Epoch, ctx: Context) -> None: ...
