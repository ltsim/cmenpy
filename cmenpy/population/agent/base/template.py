import typing

from cmenpy.hints import NDArrayType


class AgentTemplate(typing.Protocol):
    id: int
    solution: NDArrayType
    fitness: float
