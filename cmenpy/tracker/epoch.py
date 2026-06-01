import dataclasses

from cmenpy.agent import Agent


@dataclasses.dataclass
class EpochHistory:
    best: Agent
    worst: Agent
    population: list[Agent]
    timeit: int | float
