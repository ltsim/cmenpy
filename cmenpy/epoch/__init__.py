import time

from cmenpy.tracker import Tracker


class Epoch:
    def __init__(self, e: int):
        self.__e: int = e

    def __int__(self):
        return self.__e

    def __float__(self):
        return float(self.__e)

    def __repr__(self):
        return f"Epoch({self.__e})"


class EpochIteration:
    def __init__(
        self,
        epochs: int,
        tracker: Tracker,
        debug: bool = False,
    ):
        if epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")

        self.__epochs: int = epochs
        self.__current_epoch: int = 0
        self.__cpu_time: float = 0
        self.__tracker: Tracker = tracker
        self.__debug: bool = debug

    def __iter__(self):
        for e in range(0, self.__epochs):
            self.__current_epoch = e

            start = time.process_time()
            yield e
            self.__cpu_time = time.process_time() - start

            if self.__debug:
                self.__tracker.track(e, self.cpu_time)

    @property
    def current(self):
        return self.__current_epoch

    @property
    def max(self) -> int:
        return self.__epochs

    @property
    def cpu_time(self):
        return self.__cpu_time
