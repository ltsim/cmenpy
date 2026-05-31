import typing

from cmenpy.agent import Agent

A = typing.TypeVar("A", bound=Agent)

def sorted_population(agents: list[A]) -> list[A]:
    if not len(agents) > 0:
        raise ValueError("The population is empty.")

    return sorted(agents, key=lambda a: a.fitness)
