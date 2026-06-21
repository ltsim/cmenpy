import abc
import typing

import numpy as np

from cmenpy.hints import ArrayFloatType, ScalarType
from cmenpy.hints.array import ArrayObjectType, ArrayIntegerType

_DEFAULT_EPSILON_DECODER: typing.Final[float] = 1e-4

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


class BaseDecoder(typing.Generic[R], abc.ABC):
    epsilon: typing.Final[float] = _DEFAULT_EPSILON_DECODER

    name: str
    n_vars: int
    low: ArrayFloatType
    up: ArrayFloatType
    generator: np.random.Generator

    def __init__(
        self,
        name: str = "decoder",
        generator: typing.Optional[np.random.Generator] = None,
        seed: typing.Optional[int] = None,
    ) -> None:
        self.name: str = name
        self.__generator: np.random.Generator = (
            generator if generator is not None else np.random.default_rng()
        )

        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: typing.Optional[int]) -> None:
        self.__generator = np.random.default_rng(seed)

    @property
    def generator(self):
        return self.__generator

    @staticmethod
    def round_half_up(
        x: typing.Union[ArrayFloatType, typing.Sequence[float]],
    ) -> ArrayIntegerType:
        arr: ArrayFloatType = np.asarray(x, dtype=np.float64)
        frac: ArrayFloatType = arr - np.floor(arr)
        return np.where(frac < 0.5, np.floor(arr), np.ceil(arr)).astype(np.int_)

    @abc.abstractmethod
    def generate(self) -> R:
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, values: typing.Any) -> ArrayFloatType:
        raise NotImplementedError

    @abc.abstractmethod
    def correct(self, x: ArrayFloatType) -> typing.Any:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, x: ArrayFloatType) -> R:
        raise NotImplementedError
