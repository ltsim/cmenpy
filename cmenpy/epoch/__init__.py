import time
import typing

from cmenpy.target import TargetFunction
from cmenpy.tracker import Tracker


class EpochIteration:
    """Epoch counter you iterate over; tracks per-epoch CPU time and NFE.

    An ``EpochIteration`` is the ``epoch`` argument passed to a
    ``@cm.declare`` function (and the object behind ``ctx.epoch`` /
    ``ctx.epochs``). Iterating it yields ``0 .. epochs - 1`` and, when
    ``debug`` is enabled, records a `EpochHistory`
    snapshot after each epoch.

    Parameters
    ----------
    epochs : int
        Total number of epochs; must be positive.
    tracker : cmenpy.tracker.Tracker
        Recorder populated each epoch when ``debug`` is true.
    target : cmenpy.target.TargetFunction
        Objective wrapper used to read the function-evaluation count (NFE).
    debug : bool, optional
        Record a history snapshot per epoch. Default ``False``.

    Attributes
    ----------
    current : int
        Index of the epoch most recently yielded.
    max : int
        Total number of epochs.
    cpu_time : float
        Wall CPU time spent in the most recent epoch body.
    """

    def __init__(
        self,
        epochs: int,
        tracker: Tracker,
        target: TargetFunction,
        debug: bool = False,
    ):
        if epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")

        self.__epochs: int = epochs
        self.__current_epoch: int = 0
        self.__cpu_time: float = 0
        self.__tracker: Tracker = tracker
        self.__target: TargetFunction = target
        self.__debug: bool = debug

    def __iter__(self) -> typing.Iterator[int]:
        for e in range(self.__epochs):
            self.__current_epoch = e

            start = time.process_time()
            yield e
            self.__cpu_time = time.process_time() - start

            if self.__debug:
                self.__tracker.track(e, self.cpu_time, self.__target.nfe)

    @property
    def current(self) -> int:
        return self.__current_epoch

    @property
    def max(self) -> int:
        return self.__epochs

    @property
    def cpu_time(self) -> float:
        return self.__cpu_time
