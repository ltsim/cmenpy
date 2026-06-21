import typing

import numpy as np

from cmenpy.decoder.base import BaseDecoder, BoundLike, as_bound_pair
from cmenpy.hints import ArrayFloatType, ArrayIntegerType


class IntegerDecoder(BaseDecoder[ArrayIntegerType]):
    def __init__(
        self,
        low: BoundLike = -10,
        up: BoundLike = 10,
        name: str = "integer",
        generator: typing.Optional[np.random.Generator] = None,
        seed: typing.Optional[int] = None,
    ) -> None:
        super().__init__(name, generator=generator, seed=seed)

        raw_low, raw_up = as_bound_pair(low, up)

        self.low = raw_low.astype(np.int_) - 0.5
        self.up = raw_up.astype(np.int_) + 0.5 - self.epsilon
        self.n_vars = self.low.size

    def generate(self) -> ArrayIntegerType:
        low: ArrayIntegerType = (self.low + 0.5).astype(np.int_)
        high: ArrayIntegerType = (self.up + 0.5 + self.epsilon).astype(np.int_)
        return self.generator.integers(low, high + 1)

    def encode(
        self, values: typing.Union[typing.Sequence[int], ArrayIntegerType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return self.round_half_up(np.clip(x, self.low, self.up))

    def decode(self, x: ArrayFloatType) -> ArrayIntegerType:
        return self.correct(x)
