import dataclasses

from cmenpy.population.agent import Agent


@dataclasses.dataclass(frozen=True)
class EpochHistory:
    epoch: int
    best: Agent
    worst: Agent
    all: list[Agent]
    timeit: int | float

    def __hash__(self) -> int:
        return hash(self.epoch)

    def __repr__(self) -> str:
        return f"EpochHistory(epoch={self.epoch}, timeit={self.timeit})"
