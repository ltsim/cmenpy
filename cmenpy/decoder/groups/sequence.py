import typing

import numpy as np

from cmenpy.decoder.base import BaseDecoder
from cmenpy.hints import ArrayFloatType, ArrayIntegerType

R = typing.TypeVar("R")


class SequenceDecoder(BaseDecoder[R]):
    sequences: typing.Tuple[typing.Any, ...]
    index: typing.Dict[typing.Any, int]

    def __init__(
        self,
        valid_sets: typing.Sequence[typing.Any],
        return_type: typing.Callable[[typing.Any], R] = typing.cast(
            "typing.Callable[[typing.Any], R]", tuple
        ),
        name: str = "sequence",
        generator: typing.Optional[np.random.Generator] = None,
        seed: typing.Optional[int] = None,
    ) -> None:
        super().__init__(name, generator=generator, seed=seed)

        self.return_type = return_type
        self.sequences = tuple(tuple(seq) for seq in valid_sets)
        self.index = {seq: i for i, seq in enumerate(self.sequences)}
        self.n_vars = 1
        self.low = np.zeros(1, dtype=np.float64)
        self.up = np.array([len(self.sequences) - self.epsilon], dtype=np.float64)

    def generate(self) -> R:
        choice = int(self.generator.integers(0, len(self.sequences)))
        return self.return_type(self.sequences[choice])

    def encode(self, values: typing.Any) -> ArrayFloatType:
        return np.array([self.index[tuple(values)]], dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return np.array(np.clip(x, self.low, self.up), dtype=np.int_)

    def decode(self, x: ArrayFloatType) -> R:
        idx: ArrayIntegerType = self.correct(x)
        return self.return_type(self.sequences[int(idx[0])])
