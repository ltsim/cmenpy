import typing

from cmenpy.context import Context
from cmenpy.epoch import Epoch


@typing.runtime_checkable
class ModelProtocol(typing.Protocol):
    def initialize(self, ctx: Context) -> None: ...

    def evolve(self, e: Epoch, ctx: Context) -> None: ...
