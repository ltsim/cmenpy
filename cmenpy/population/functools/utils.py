import typing

from cmenpy.agent import Agent


def sorted_population(agents: list[Agent]) -> list[Agent]:
    if not len(agents) > 0:
        raise ValueError("The population is empty.")

    return sorted(agents, key=lambda a: a.fitness)
