import typing

import numpy as np

from cmenpy.hints import ArrayFloatType, ScalarType
from cmenpy.hints.array import ArrayObjectType, ArrayIntegerType, ArrayBoolType

BoundLike = typing.Union[ScalarType, typing.Sequence[ScalarType]]

R = typing.TypeVar("R")


def as_bound_pair(
    lb: BoundLike, ub: BoundLike
) -> typing.Tuple[ArrayFloatType, ArrayFloatType]:
    if isinstance(lb, (int, float)) and isinstance(ub, (int, float)):
        return np.array([float(lb)], dtype=np.float64), np.array(
            [float(ub)], dtype=np.float64
        )
    if isinstance(lb, typing.Sequence) and isinstance(ub, typing.Sequence):
        if len(lb) != len(ub):
            raise ValueError("lb and ub must have the same length.")
        return np.array(lb, dtype=np.float64), np.array(ub, dtype=np.float64)
    raise TypeError("lb and ub must both be scalars or both be sequences.")


def as_label_set(
    valid_set: typing.Sequence[typing.Any],
) -> typing.Tuple[ArrayObjectType, typing.Dict[typing.Any, int]]:
    arr: ArrayObjectType = np.array(sorted(valid_set, key=str), dtype=object)
    table = {label: i for i, label in enumerate(arr)}
    return arr, table


def as_label_sets(
    valid_sets: typing.Sequence[typing.Any],
) -> typing.Tuple[
    typing.Tuple[ArrayObjectType, ...], typing.Tuple[typing.Dict[typing.Any, int], ...]
]:
    first = valid_sets[0]
    if not isinstance(first, (tuple, list, np.ndarray)):
        groups: typing.Tuple[typing.Tuple[typing.Any, ...], ...] = (tuple(valid_sets),)
    else:
        for vset in valid_sets:
            if len(vset) < 2:
                raise ValueError("Each set must contain at least two values.")
        groups = tuple(tuple(vset) for vset in valid_sets)

    pairs = [as_label_set(group) for group in groups]
    labels = tuple(arr for arr, _ in pairs)
    index = tuple(table for _, table in pairs)
    return labels, index


class BaseSpaceDecoder(typing.Generic[R]):
    EPS: float = 1e-4

    name: str
    n_vars: int
    lb: ArrayFloatType
    ub: ArrayFloatType
    generator: np.random.Generator

    def __init__(self, name: str = "decoder") -> None:
        self.name = name
        self.generator = np.random.default_rng()

    def set_seed(self, seed: typing.Optional[int]) -> None:
        self.generator = np.random.default_rng(seed)

    @staticmethod
    def round_half_up(
        x: typing.Union[ArrayFloatType, typing.Sequence[float]],
    ) -> ArrayIntegerType:
        arr: ArrayFloatType = np.asarray(x, dtype=np.float64)
        frac: ArrayFloatType = arr - np.floor(arr)
        return np.where(frac < 0.5, np.floor(arr), np.ceil(arr)).astype(np.int_)

    def generate(self) -> R:
        raise NotImplementedError

    def encode(self, values: typing.Any) -> ArrayFloatType:
        raise NotImplementedError

    def correct(self, x: ArrayFloatType) -> typing.Any:
        raise NotImplementedError

    def decode(self, x: ArrayFloatType) -> R:
        raise NotImplementedError


class FloatSpaceDecoder(BaseSpaceDecoder[ArrayFloatType]):
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


class IntegerSpaceDecoder(BaseSpaceDecoder[ArrayIntegerType]):
    def __init__(
        self, lb: BoundLike = -10, ub: BoundLike = 10, name: str = "integer"
    ) -> None:
        super().__init__(name)
        raw_lb, raw_ub = as_bound_pair(lb, ub)
        self.lb = raw_lb.astype(np.int_) - 0.5
        self.ub = raw_ub.astype(np.int_) + 0.5 - self.EPS
        self.n_vars = self.lb.size

    def generate(self) -> ArrayIntegerType:
        low: ArrayIntegerType = (self.lb + 0.5).astype(np.int_)
        high: ArrayIntegerType = (self.ub + 0.5 + self.EPS).astype(np.int_)
        return self.generator.integers(low, high + 1)

    def encode(
        self, values: typing.Union[typing.Sequence[int], ArrayIntegerType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return self.round_half_up(np.clip(x, self.lb, self.ub))

    def decode(self, x: ArrayFloatType) -> ArrayIntegerType:
        return self.correct(x)


class StringSpaceDecoder(BaseSpaceDecoder[typing.List[typing.Any]]):
    labels: typing.Tuple[ArrayObjectType, ...]
    index: typing.Tuple[typing.Dict[typing.Any, int], ...]

    def __init__(
        self, valid_sets: typing.Sequence[typing.Any], name: str = "string"
    ) -> None:
        super().__init__(name)
        self.labels, self.index = as_label_sets(valid_sets)
        self.n_vars = len(self.labels)
        self.lb = np.zeros(self.n_vars, dtype=np.float64)
        self.ub = np.array(
            [len(arr) - self.EPS for arr in self.labels], dtype=np.float64
        )

    def generate(self) -> typing.List[typing.Any]:
        return [self.generator.choice(arr) for arr in self.labels]

    def encode(self, values: typing.Sequence[typing.Any]) -> ArrayFloatType:
        return np.array(
            [table[val] for table, val in zip(self.index, values)], dtype=np.float64
        )

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return np.array(np.clip(x, self.lb, self.ub), dtype=np.int_)

    def decode(self, x: ArrayFloatType) -> typing.List[typing.Any]:
        idx: ArrayIntegerType = self.correct(x)
        return [arr[i] for arr, i in zip(self.labels, idx)]


class CategoricalSpaceDecoder(StringSpaceDecoder):
    def generate(self) -> typing.List[typing.Any]:
        return [self.generator.choice(arr) for arr in self.labels]


class SequenceSpaceDecoder(BaseSpaceDecoder[R]):
    sequences: typing.Tuple[typing.Any, ...]
    index: typing.Dict[typing.Any, int]

    def __init__(
        self,
        valid_sets: typing.Sequence[typing.Any],
        return_type: typing.Callable[[typing.Any], R] = typing.cast(
            "typing.Callable[[typing.Any], R]", tuple
        ),
        name: str = "sequence",
    ) -> None:
        super().__init__(name)
        self.return_type = return_type
        self.sequences = tuple(tuple(seq) for seq in valid_sets)
        self.index = {seq: i for i, seq in enumerate(self.sequences)}
        self.n_vars = 1
        self.lb = np.zeros(1, dtype=np.float64)
        self.ub = np.array([len(self.sequences) - self.EPS], dtype=np.float64)

    def generate(self) -> R:
        choice = int(self.generator.integers(0, len(self.sequences)))
        return self.return_type(self.sequences[choice])

    def encode(self, values: typing.Any) -> ArrayFloatType:
        return np.array([self.index[tuple(values)]], dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return np.array(np.clip(x, self.lb, self.ub), dtype=np.int_)

    def decode(self, x: ArrayFloatType) -> R:
        idx: ArrayIntegerType = self.correct(x)
        return self.return_type(self.sequences[int(idx[0])])


class PermutationSpaceDecoder(BaseSpaceDecoder[ArrayObjectType]):
    labels: ArrayObjectType
    index: typing.Dict[typing.Any, int]

    def __init__(
        self, valid_set: typing.Sequence[typing.Any], name: str = "permutation"
    ) -> None:
        super().__init__(name)
        if len(valid_set) < 2:
            raise ValueError("Permutation needs at least two elements.")
        self.labels, self.index = as_label_set(valid_set)
        self.n_vars = len(self.labels)
        self.lb = np.zeros(self.n_vars, dtype=np.float64)
        self.ub = np.full(self.n_vars, self.n_vars - self.EPS, dtype=np.float64)

    def generate(self) -> ArrayObjectType:
        return self.generator.permutation(self.labels)

    def encode(self, values: typing.Sequence[typing.Any]) -> ArrayFloatType:
        return np.array([self.index[val] for val in values], dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return np.argsort(x).astype(np.int_)

    def decode(self, x: ArrayFloatType) -> ArrayObjectType:
        order: ArrayIntegerType = self.correct(x)
        return self.labels[order]


class BinarySpaceDecoder(BaseSpaceDecoder[ArrayIntegerType]):
    def __init__(self, n_vars: int = 1, name: str = "binary") -> None:
        super().__init__(name)
        if not isinstance(n_vars, int) or n_vars < 1:
            raise ValueError("n_vars must be a positive integer.")
        self.n_vars = n_vars
        self.lb = np.zeros(n_vars, dtype=np.float64)
        self.ub = np.full(n_vars, 2.0 - self.EPS, dtype=np.float64)

    def generate(self) -> ArrayIntegerType:
        return self.generator.integers(0, 2, self.n_vars)

    def encode(
        self, values: typing.Union[typing.Sequence[int], ArrayIntegerType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return np.array(np.clip(x, self.lb, self.ub), dtype=np.int_)

    def decode(self, x: ArrayFloatType) -> ArrayIntegerType:
        return self.correct(x)


class BoolSpaceDecoder(BaseSpaceDecoder[ArrayBoolType]):
    def __init__(self, n_vars: int = 1, name: str = "bool") -> None:
        super().__init__(name)
        if not isinstance(n_vars, int) or n_vars < 1:
            raise ValueError("n_vars must be a positive integer.")
        self.n_vars = n_vars
        self.lb = np.zeros(n_vars, dtype=np.float64)
        self.ub = np.full(n_vars, 2.0 - self.EPS, dtype=np.float64)

    def generate(self) -> ArrayBoolType:
        return self.generator.integers(0, 2, self.n_vars).astype(np.bool_)

    def encode(
        self, values: typing.Union[typing.Sequence[bool], ArrayBoolType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayBoolType:
        clipped: ArrayFloatType = np.clip(x, self.lb, self.ub)
        return np.array(clipped, dtype=np.int_).astype(np.bool_)

    def decode(self, x: ArrayFloatType) -> ArrayBoolType:
        return self.correct(x)
