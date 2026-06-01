import typing

from cmenpy.tracker.epoch import EpochHistory


class Tracker:
    def __init__(self, history: typing.Collection[EpochHistory]):
        self.___history = history

    @property
    def history(self) -> list[EpochHistory]:
        return [*self.___history]
