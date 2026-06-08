from cmenpy.population.people import PeopleSequenceCollection, PeopleGenerator


def sort_agents(
    population: PeopleGenerator,
) -> PeopleSequenceCollection:
    return population.agents.sort
