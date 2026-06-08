import typing

from cmenpy.functools.utils import sort_agents
from cmenpy.population.agent import Agent
from cmenpy.types.option import SenseType


def is_best(ab: typing.Any, sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a[0] < b[0]

    return a[0] > b[0]


def is_worst(ab: typing.Any, sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a[0] > b[0]

    return a[0] < b[0]


def best_of(agents: typing.Collection[Agent], sense: SenseType = "min") -> Agent:
    if sense == "min":
        return min(agents)

    return max(agents)


def worst_of(agents: typing.Collection[Agent], sense: SenseType = "min") -> Agent:
    if sense == "max":
        return min(agents)

    return max(agents)


__all__ = ["sort_agents", "is_best", "is_worst", "best_of", "worst_of"]
