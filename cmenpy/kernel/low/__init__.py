import numpy as np

from cmenpy.hints import NDArrayType


def init_buffer(
    n_pop: int,
    ndim: int,
) -> NDArrayType:
    return np.full((n_pop, ndim + 1), np.nan)
