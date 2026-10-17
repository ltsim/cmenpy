"""Agent protocol: identity, solution and fitness of an individual."""

import typing

from cmenpy.hints.array import NDArrayType


class AgentTemplate(typing.Protocol):
    """Protocol describing an agent: identity, solution and fitness."""

    id: int
    solution: NDArrayType
    fitness: float
