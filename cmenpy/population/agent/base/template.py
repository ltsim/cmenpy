import typing

from cmenpy.hints import ArrayType


class AgentTemplate(typing.Protocol):
    id: int
    solution: ArrayType
    fitness: float
