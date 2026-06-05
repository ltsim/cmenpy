from cmenpy.population import PopulationGenerator
from cmenpy.population.agents import AgentSequenceCollection


def sort_agents(
    population: PopulationGenerator,
) -> AgentSequenceCollection:
    return population.agents.sort
