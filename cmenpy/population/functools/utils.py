from cmenpy.population import PeopleGenerator
from cmenpy.population.people import PeopleSequenceCollection


def sort_agents(
    population: PeopleGenerator,
) -> PeopleSequenceCollection:
    return population.agents.sort
