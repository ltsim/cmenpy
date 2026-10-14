import typing

from cmenpy.context import Context


@typing.runtime_checkable
class ModelProtocol(typing.Protocol):
    """Protocol a ``@cm.template`` class must satisfy.

    The class must implement `initialize` (seed the population once)
    and `evolve` (advance one epoch).
    """

    def initialize(self, ctx: Context) -> None: ...

    def evolve(self, e: int, ctx: Context) -> None: ...
