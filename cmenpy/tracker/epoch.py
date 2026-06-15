import dataclasses

from cmenpy.hints import ArrayType


@dataclasses.dataclass(frozen=True)
class EpochHistory:
    epoch: int
    best_idx: int
    worst_idx: int
    all: ArrayType
    timeit: int | float
    nfe: int

    def __hash__(self) -> int:
        return hash(self.epoch)

    def __repr__(self) -> str:
        return f"EpochHistory(epoch={self.epoch}, timeit={self.timeit})"
