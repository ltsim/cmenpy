import collections
import typing

from cmenpy.population import Population
from cmenpy.tracker.epoch import EpochHistory


class Tracker:
    def __init__(self, population: Population):
        self.___history = collections.deque()
        self.___population = population

    @property
    def history(self) -> list[EpochHistory]:
        return [*self.___history]

    def track(self) -> None: ...
