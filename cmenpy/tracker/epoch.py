"""Immutable per-epoch snapshot (`EpochHistory`)."""

import dataclasses

from cmenpy.hints import NDArrayType


@dataclasses.dataclass(frozen=True)
class EpochHistory:
    """Immutable snapshot of one epoch (only recorded when ``debug=True``).

    Attributes:
        epoch: Epoch index.
        best_idx: Row index of the best individual at the end of the epoch.
        worst_idx: Row index of the worst individual at the end of the epoch.
        all: Copy of the full population buffer (fitness in column 0).
        timeit: CPU time spent in the epoch.
        nfe: Function evaluations spent up to the end of the epoch.
    """

    epoch: int
    best_idx: int
    worst_idx: int
    all: NDArrayType
    timeit: int | float
    nfe: int

    def __hash__(self) -> int:
        """Hash by the epoch index."""
        return hash(self.epoch)

    def __repr__(self) -> str:
        """Return a compact ``EpochHistory(epoch=..., timeit=...)`` rendering."""
        return f"EpochHistory(epoch={self.epoch}, timeit={self.timeit})"
