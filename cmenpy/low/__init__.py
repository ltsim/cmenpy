import numpy as np

from cmenpy.types import NDArrayType


def expand_buffer(
    n: int,
    ndim: int,
) -> NDArrayType:
    return np.hstack(([n], np.full((ndim + 1), np.nan)))


def init_buffer(
    n_pop: int,
    ndim: int,
) -> NDArrayType:
    return np.full((n_pop, ndim + 1), np.nan)
