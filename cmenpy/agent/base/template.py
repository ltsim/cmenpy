import typing

from cmenpy.types import NDArrayType


class AgentTemplate(typing.Protocol):
    id: int
    solution: NDArrayType
    fitness: float
