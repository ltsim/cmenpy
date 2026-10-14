import typing

from cmenpy.hints import NDArrayType


class AgentTemplate(typing.Protocol):
    """Protocol describing an agent: identity, solution and fitness."""

    id: int
    solution: NDArrayType
    fitness: float
