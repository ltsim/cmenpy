import typing

import numpy as np

from cmenpy.decoder.base import BaseDecoder, BoundLike, as_bound_pair
from cmenpy.hints import ArrayFloatType


class FloatingDecoder(BaseDecoder[ArrayFloatType]):
    def __init__(
        self,
        low: BoundLike = -10.0,
        up: BoundLike = 10.0,
        name: str = "float",
        generator: typing.Optional[np.random.Generator] = None,
        seed: typing.Optional[int] = None,
    ) -> None:
        super().__init__(name, generator=generator, seed=seed)

        self.low, self.up = as_bound_pair(low, up)
        self.n_vars = self.low.size

    def generate(self) -> ArrayFloatType:
        return self.generator.uniform(self.low, self.up)

    def encode(
        self, values: typing.Union[typing.Sequence[float], ArrayFloatType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayFloatType:
        return np.clip(x, self.low, self.up)

    def decode(self, x: ArrayFloatType) -> ArrayFloatType:
        return self.correct(x)
