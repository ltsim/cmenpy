"""Explicit casting of population views (`cm.cast`) and the deprecated hook."""

import warnings

import numpy as np
import pytest

import cmenpy as cm
from cmenpy.target import TargetFunction

BOUNDS = cm.Bounds[(-1, 1), (-1, 1)]


def _population(size: int = 4) -> cm.PopulationSwarm:
    """Return a scored swarm over a fixed ``size`` x 2 block."""
    target = TargetFunction(lambda x: float(np.sum(np.square(x))), BOUNDS)
    buff = cm.KernelBuffer(cm.KernelSize(size), BOUNDS, target, "min")
    pop = cm.PopulationSwarm(buff)

    pop.compute << np.arange(size * 2, dtype=float).reshape(size, 2)

    return pop


def test_cast_view_returns_array() -> None:
    pop = _population()
    array = cm.cast(pop.X)

    assert isinstance(array, np.ndarray)
    assert array.shape == (4, 2)


def test_cast_view_is_a_live_view() -> None:
    pop = _population()
    array = cm.cast(pop.X)

    array[0, 0] = 42.0

    assert pop.X[0, 0] == 42.0


def test_cast_applies_dtype() -> None:
    pop = _population()
    array = cm.cast(pop.X, np.float32)

    assert array.dtype == np.float32


def test_cast_forces_contiguous_layout() -> None:
    pop = _population()
    assert not cm.cast(pop.X).flags["C_CONTIGUOUS"]

    contiguous = cm.cast(pop.X, float, layout="C")

    assert contiguous.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(contiguous, np.asarray(pop.X.to_numpy()))


def test_cast_scalar_to_dtype_scalar() -> None:
    assert cm.cast(1, np.float32) == np.float32(1.0)
    assert isinstance(cm.cast(1, np.float32), np.floating)


def test_cast_accepts_lists_and_arrays() -> None:
    np.testing.assert_allclose(cm.cast([1, 2, 3], float), [1.0, 2.0, 3.0])
    array = np.arange(4.0)

    assert cm.cast(array) is array


def test_to_numpy_matches_view_contents() -> None:
    pop = _population()

    np.testing.assert_allclose(pop.X.to_numpy(), pop.X[: pop.size])
    np.testing.assert_allclose(pop.F.to_numpy(), pop.F[: pop.size])


def test_cast_does_not_trigger_the_array_hook() -> None:
    pop = _population()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        array = cm.cast(pop.F, float)

    assert array.dtype == np.float64


def test_implicit_casting_is_deprecated() -> None:
    pop = _population()

    with pytest.warns(DeprecationWarning, match="implicit casting"):
        array = np.asarray(pop.X)

    assert array.shape == (4, 2)


def test_deprecated_cast_honours_dtype_and_copy() -> None:
    pop = _population()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        upcast = np.asarray(pop.X, dtype=np.float32)
        copied = np.array(pop.X, copy=True)

    assert upcast.dtype == np.float32
    assert copied.base is None


def test_comparators_accept_views() -> None:
    pop = _population()
    reference = np.array([0.0, 20.0, 50.0, 0.0])

    # The live fitness view is unwrapped internally, not via ``__array__``.
    improved = cm.is_best((reference, pop.F), "min")

    assert improved.tolist() == [0, 3]
