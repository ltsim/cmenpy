import numpy as np

import cmenpy as cm


def test_best_index_min() -> None:
    values = np.array([3.0, 1.0, 2.0])

    assert cm.best_index(values, "min") == 1
    assert cm.worst_index(values, "min") == 0


def test_best_index_max() -> None:
    values = np.array([3.0, 1.0, 2.0])

    assert cm.best_index(values, "max") == 0
    assert cm.worst_index(values, "max") == 1


def test_best_index_accepts_list() -> None:
    assert cm.best_index([5, 2, 9], "max") == 2
    assert cm.worst_index([5, 2, 9], "max") == 1
