import numpy as np

from cmenpy.hints import NDArrayType


def init_buffer(
    n_pop: int,
    ndim: int,
) -> NDArrayType:
    """Allocate a ``(n_pop, ndim + 1)`` buffer filled with ``NaN``.

    Parameters
    ----------
    n_pop : int
        Number of rows (population capacity).
    ndim : int
        Number of solution dimensions (the +1 column holds fitness).

    Returns
    -------
    NDArrayType
        The freshly allocated ``NaN`` buffer.
    """
    return np.full((n_pop, ndim + 1), np.nan)
