import numpy as np

from cmenpy.hints import ArrayType


def expand_buffer(
    n: int,
    ndim: int,
) -> ArrayType:
    return np.hstack(([n], np.full((ndim + 1), np.nan)))


def init_buffer(
    n_pop: int,
    ndim: int,
) -> ArrayType:
    return np.full((n_pop, ndim + 1), np.nan)
