import collections

from cmenpy.kernel import KernelBuffer
from cmenpy.tracker.epoch import EpochHistory


class Tracker:
    """Records per-epoch snapshots of the population buffer.

    Parameters
    ----------
    buff : KernelBuffer
        The buffer whose state is snapshotted on every `track`.

    Attributes
    ----------
    history : list[EpochHistory]
        Immutable copies of the tracked epochs, oldest first.
    """

    def __init__(self, buff: KernelBuffer):
        self.___history: collections.deque[EpochHistory] = collections.deque()
        self.__k_buff = buff

    @property
    def history(self) -> list[EpochHistory]:
        return [*self.___history]

    def track(self, e: int, timeit: float, nfe: int) -> None:
        self.___history.append(
            EpochHistory(
                epoch=e,
                best_idx=int(self.__k_buff.idx.best),
                worst_idx=int(self.__k_buff.idx.worst),
                all=self.__k_buff.raw.copy(),
                timeit=timeit,
                nfe=nfe,
            )
        )


__all__ = ["EpochHistory", "Tracker"]
