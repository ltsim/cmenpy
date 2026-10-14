import dataclasses

from cmenpy.hints import NDArrayType


@dataclasses.dataclass(frozen=True)
class EpochHistory:
    """Immutable snapshot of one epoch (only recorded when ``debug=True``).

    Attributes
    ----------
    epoch : int
        Epoch index.
    best_idx : int
        Row index of the best individual at the end of the epoch.
    worst_idx : int
        Row index of the worst individual at the end of the epoch.
    all : NDArrayType
        Copy of the full population buffer (fitness in column 0).
    timeit : int or float
        CPU time spent in the epoch.
    nfe : int
        Function evaluations spent up to the end of the epoch.
    """

    epoch: int
    best_idx: int
    worst_idx: int
    all: NDArrayType
    timeit: int | float
    nfe: int

    def __hash__(self) -> int:
        return hash(self.epoch)

    def __repr__(self) -> str:
        return f"EpochHistory(epoch={self.epoch}, timeit={self.timeit})"
