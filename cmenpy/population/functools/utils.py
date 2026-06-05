from cmenpy.population import PopulationGenerator
from cmenpy.population.people import PeopleSequenceCollection


def sort_agents(
    population: PopulationGenerator,
) -> PeopleSequenceCollection:
    return population.agents.sort
