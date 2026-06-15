import typing

import numpy as np

from cmenpy.decoder.base import BaseDecoder, BoundLike, as_bound_pair
from cmenpy.hints import ArrayFloatType, ArrayIntegerType


class IntegerDecoder(BaseDecoder[ArrayIntegerType]):
    def __init__(
        self, lb: BoundLike = -10, ub: BoundLike = 10, name: str = "integer"
    ) -> None:
        super().__init__(name)
        raw_lb, raw_ub = as_bound_pair(lb, ub)
        self.lb = raw_lb.astype(np.int_) - 0.5
        self.ub = raw_ub.astype(np.int_) + 0.5 - self.epsilon
        self.n_vars = self.lb.size

    def generate(self) -> ArrayIntegerType:
        low: ArrayIntegerType = (self.lb + 0.5).astype(np.int_)
        high: ArrayIntegerType = (self.ub + 0.5 + self.epsilon).astype(np.int_)
        return self.generator.integers(low, high + 1)

    def encode(
        self, values: typing.Union[typing.Sequence[int], ArrayIntegerType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return self.round_half_up(np.clip(x, self.lb, self.ub))

    def decode(self, x: ArrayFloatType) -> ArrayIntegerType:
        return self.correct(x)
