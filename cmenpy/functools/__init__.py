import typing

from cmenpy.functools.utils import sort_agents
from cmenpy.population.agent import Agent
from cmenpy.types.option import SenseType

A = typing.TypeVar("A")
B = typing.TypeVar("B")
AB = tuple[A, B]


def is_best(ab: AB, sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a < b

    return a > b


def is_worst(ab: AB, sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a > b

    return a < b


def best_of(agents: typing.Collection[A], sense: SenseType = "min") -> Agent:
    if sense == "min":
        return min(agents)

    return max(agents)


def worst_of(agents: typing.Collection[A], sense: SenseType = "min") -> Agent:
    if sense == "max":
        return min(agents)

    return max(agents)


__all__ = ["sort_agents", "is_best", "is_worst", "best_of", "worst_of"]
