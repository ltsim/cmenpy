from cmenpy.agent import Agent
from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType

AgentIterator = Agent | NDArrayType


def is_best(ab: tuple[AgentIterator, AgentIterator], sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a[0] < b[0]

    return a[0] > b[0]


def is_worst(ab: tuple[AgentIterator, AgentIterator], sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a[0] > b[0]

    return a[0] < b[0]
