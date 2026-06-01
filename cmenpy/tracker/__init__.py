import collections

from cmenpy.population import BufferPopulation
from cmenpy.tracker.epoch import EpochHistory


class Tracker:
    def __init__(self, population: BufferPopulation):
        self.___history = collections.deque()
        self.___population = population

    @property
    def history(self) -> list[EpochHistory]:
        return [*self.___history]

    def track(self, e: int, timeit: int | float) -> None:
        self.___history.append(
            EpochHistory(
                epoch=e,
                best=self.___population.best,
                worst=self.___population.worst,
                all=self.___population.all,
                timeit=timeit,
            )
        )
