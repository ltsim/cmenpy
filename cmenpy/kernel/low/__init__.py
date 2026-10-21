"""Low-level allocation helpers for the kernel buffer."""

import numpy as np

from cmenpy.hints import NDArrayType


def init_buffer(
    n_pop: int,
    ndim: int,
) -> NDArrayType:
    """Allocate a ``(n_pop, ndim + 1)`` buffer filled with ``NaN``.

    Args:
        n_pop: Number of rows (population capacity).
        ndim: Number of solution dimensions (the +1 column holds fitness).

    Returns:
        The freshly allocated ``NaN`` buffer.
    """
    return np.full((n_pop, ndim + 1), np.nan)
