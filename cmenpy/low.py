import numpy as np


def expand_buffer(n: int, ndim: int) -> np.ndarray:
    return np.hstack(([n], np.full((ndim + 1), np.nan)))


def init_buffer(n_pop: int, ndim: int) -> np.ndarray:
    return np.full((n_pop, ndim + 1), np.nan)
