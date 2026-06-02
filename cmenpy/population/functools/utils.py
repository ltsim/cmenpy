from cmenpy.agent import Agent
from cmenpy.types.option import SenseType


def sort_agents(agents: list[Agent], sense: SenseType = "min") -> list[Agent]:
    if not len(agents) > 0:
        raise ValueError("The population is empty.")

    sorted_agents = sorted(agents, key=lambda a: a.fitness)

    return sorted_agents if sense == "min" else sorted_agents[::-1]
