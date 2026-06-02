import typing

from cmenpy.agent import Agent


class IterableAgent(typing.Protocol):
    def __len__(self) -> int: ...

    def __iter__(self) -> typing.Iterator[Agent]: ...
