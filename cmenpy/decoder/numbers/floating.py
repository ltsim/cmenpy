import typing

import numpy as np

from cmenpy.decoder.base import BaseDecoder, BoundLike, as_bound_pair
from cmenpy.hints import ArrayFloatType


class FloatingDecoder(BaseDecoder[ArrayFloatType]):
    def __init__(
        self, lb: BoundLike = -10.0, ub: BoundLike = 10.0, name: str = "float"
    ) -> None:
        super().__init__(name)
        self.lb, self.ub = as_bound_pair(lb, ub)
        self.n_vars = self.lb.size

    def generate(self) -> ArrayFloatType:
        return self.generator.uniform(self.lb, self.ub)

    def encode(
        self, values: typing.Union[typing.Sequence[float], ArrayFloatType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayFloatType:
        return np.clip(x, self.lb, self.ub)

    def decode(self, x: ArrayFloatType) -> ArrayFloatType:
        return self.correct(x)
