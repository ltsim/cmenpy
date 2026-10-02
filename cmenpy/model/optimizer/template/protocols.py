import typing

from cmenpy.context import Context


@typing.runtime_checkable
class ModelProtocol(typing.Protocol):
    def initialize(self, ctx: Context) -> None: ...

    def evolve(self, e: int, ctx: Context) -> None: ...
