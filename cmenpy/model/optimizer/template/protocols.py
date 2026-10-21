"""The `ModelProtocol` a ``@cm.template`` class must satisfy."""

import typing

from cmenpy.context import Context


@typing.runtime_checkable
class ModelProtocol(typing.Protocol):
    """Protocol a ``@cm.template`` class must satisfy.

    The class must implement `initialize` (seed the population once)
    and `evolve` (advance one epoch).
    """

    def initialize(self, ctx: Context) -> None:
        """Seed the population once before the first epoch."""

    def evolve(self, e: int, ctx: Context) -> None:
        """Advance the population one epoch (``e`` is the epoch index)."""
