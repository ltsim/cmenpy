"""Tests for the optional numba backend and the ``numba=`` compile switch.

These run in *both* environments: with numba installed (compilation is
controlled by the ``numba`` option) and without it (pure-Python fallbacks).
"""

import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import pytest

import cmenpy as cm
from cmenpy.compile import backend

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Box:
    """Object that only exposes the NumPy array protocol, like the buffer views."""

    def __init__(self, values: typing.Any) -> None:
        self._values = np.asarray(values)

    def __array__(
        self, dtype: typing.Any = None, copy: typing.Any = None
    ) -> typing.Any:
        return np.asarray(self._values, dtype=dtype)


_BLOCKER = """
import sys


class _BlockNumba:
    def find_spec(self, name, path=None, target=None):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba is blocked for testing")

        return None


sys.meta_path.insert(0, _BlockNumba())
"""


def _template_model() -> typing.Any:
    """Build a fresh deterministic template model with a scaling kernel."""

    @cm.template
    class Halve:
        f: cm.Argument[float, (0.0, 2.0), 0.5]

        def initialize(self, ctx) -> None:
            pop = cm.PopulationSwarm(ctx.buff)
            pop.compute << np.full((pop.size, ctx.bounds.ndim), 2.0)

        def evolve(self, e, ctx) -> None:
            pop = cm.PopulationSwarm(ctx.buff)
            pop.compute << self.scale(pop.X[: pop.size], self.f)

        @cm.precompile
        def scale(X: "Array[float, 2]", f: "Scalar[float]"):
            return X * f

    return Halve


def _sphere(x: typing.Any) -> float:
    return float(np.sum(x**2))


def test_model_numba_false_runs_uncompiled() -> None:
    model = _template_model()(seed=1, numba=False)

    assert model.compile_options["numba"] is False

    reported = model.prepare()

    assert reported["scale"] == ["<python>"]

    # One epoch over a (2,) population of 2.0s scaled by 0.5 -> x = [1, 1].
    fitness, x = model.solve(_sphere, cm.Bounds[(-1, 1), (-1, 1)], 1, 4)

    assert fitness == pytest.approx(2.0)
    assert x.shape == (2,)


def test_model_default_keeps_numba_key_absent() -> None:
    model = _template_model()(seed=1)

    assert "numba" not in model.compile_options


def test_model_numba_true_needs_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "HAS_NUMBA", False)

    with pytest.raises(ImportError, match="cmenpy needs numba"):
        _template_model()(seed=1, numba=True)


@pytest.mark.skipif(not backend.HAS_NUMBA, reason="needs the numba backend")
def test_model_numba_true_compiles() -> None:
    outer = _template_model()
    kernel = vars(outer.model_class)["scale"]

    model = outer(seed=1, numba=True)

    assert model.compile_options["numba"] is True
    assert model.prepare()["scale"] != ["<python>"]
    assert kernel._cache


def test_auto_falls_back_when_numba_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "HAS_NUMBA", False)

    model = _template_model()(seed=1)

    assert model.prepare()["scale"] == ["<python>"]

    fitness, _ = model.solve(_sphere, cm.Bounds[(-1, 1), (-1, 1)], 1, 3)

    assert fitness == pytest.approx(2.0)


def test_kernel_stays_uncompiled_when_disabled() -> None:
    outer = _template_model()
    kernel = vars(outer.model_class)["scale"]

    model = outer(seed=1, numba=False)
    model.solve(_sphere, cm.Bounds[(-1, 1), (-1, 1)], 1, 3)

    assert kernel._cache == {}


def test_decorator_numba_false_wins_over_default() -> None:
    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

        @cm.precompile(numba=False)
        def slow(X: "Array[float, 2]"):
            return X + 1.0

        @cm.precompile
        def fast(X: "Array[float, 2]"):
            return X * 2.0

    reported = Model(seed=1).prepare()

    assert reported["slow"] == ["<python>"]

    if backend.HAS_NUMBA:
        assert reported["fast"] != ["<python>"]
        assert reported["fast"]


def test_solve_prepare_flag_reports_python_mode() -> None:
    model = _template_model()(seed=1, numba=False)

    fitness, _ = model.solve(_sphere, cm.Bounds[(-1, 1), (-1, 1)], 1, 3, prepare=True)

    assert fitness == pytest.approx(2.0)


def test_vectorize_python_flavor_is_elementwise() -> None:
    @cm.vectorize(numba=False)
    def affine(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]":
        return a * r + 1.0

    np.testing.assert_allclose(affine(np.array([1.0, 2.0, 3.0]), 2.0), [3.0, 5.0, 7.0])


def test_reduce_python_flavor_matches_numpy() -> None:
    @cm.reduce("(n),(),()->(n)", numba=False)
    def clip(
        row: "Array[float, 'n']",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)

    matrix = np.array([[-5.0, 0.5, 5.0], [10.0, -10.0, 1.0]])

    np.testing.assert_allclose(clip(matrix, -1.0, 1.0), np.clip(matrix, -1.0, 1.0))


def test_reduce_python_flavor_supports_out_keyword() -> None:
    @cm.reduce(
        [cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]],
        "(n),()->(n)",
        numba=False,
    )
    def scale(row, factor, out):
        for i in range(row.shape[0]):
            out[i] = row[i] * factor

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.zeros_like(matrix)
    returned = scale(matrix, 2.0, out=target)

    np.testing.assert_allclose(target, matrix * 2.0)
    assert returned is target


def test_contract_python_flavor_matmul() -> None:
    @cm.reduce(
        cm.Contract[
            [cm.Array[float, "m,n"], cm.Array[float, "n,k"]],
            cm.Array[float, "m,k"],
        ],
        numba=False,
    )
    def matmul(a, b, out):
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                acc = 0.0

                for k in range(a.shape[1]):
                    acc += a[i, k] * b[k, j]

                out[i, j] = acc

    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)

    np.testing.assert_allclose(matmul(A, B), A @ B)


def test_contract_python_flavor_broadcasts_loop_dims() -> None:
    @cm.reduce(
        cm.Contract[
            [cm.Array[float, "b,m,n"], cm.Array[float, "b,n,k"]],
            cm.Array[float, "b,m,k"],
        ],
        numba=False,
    )
    def batch_matmul(a, b, out):
        for batch in range(a.shape[0]):
            for i in range(a.shape[1]):
                for j in range(b.shape[2]):
                    acc = 0.0

                    for k in range(a.shape[2]):
                        acc += a[batch, i, k] * b[batch, k, j]

                    out[batch, i, j] = acc

    A = np.arange(24.0).reshape(2, 3, 4)
    B = np.arange(40.0).reshape(2, 4, 5)

    np.testing.assert_allclose(batch_matmul(A, B), A @ B)


def test_prange_is_range_without_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "HAS_NUMBA", False)

    assert cm.prange is range


@pytest.mark.skipif(not backend.HAS_NUMBA, reason="needs the numba backend")
def test_prange_is_numba_prange_with_numba() -> None:
    assert cm.prange is backend.numba_module().prange


def test_import_and_run_without_numba() -> None:
    """Run the package in a fresh interpreter where numba cannot be found."""
    script = (
        "import numpy as np\n"
        "import cmenpy as cm\n"
        "assert not cm.HAS_NUMBA\n"
        "assert cm.prange is range\n"
        "@cm.template\n"
        "class Halve:\n"
        "    def initialize(self, ctx):\n"
        "        pop = cm.PopulationSwarm(ctx.buff)\n"
        "        pop.compute << np.full((pop.size, ctx.bounds.ndim), 2.0)\n"
        "    def evolve(self, e, ctx):\n"
        "        pop = cm.PopulationSwarm(ctx.buff)\n"
        "        pop.compute << self.scale(pop.X[: pop.size], 0.5)\n"
        "    @cm.precompile\n"
        "    def scale(X: 'Array[float, 2]', f: 'Scalar[float]'):\n"
        "        return X * f\n"
        "model = Halve(seed=1)\n"
        "assert model.prepare()['scale'] == ['<python>']\n"
        "f, x = model.solve(lambda v: float(np.sum(v**2)), "
        "cm.Bounds[(-1, 1), (-1, 1)], 1, 3)\n"
        "assert abs(f - 2.0) < 1e-9, f\n"
        "print('no-numba-ok')\n"
    )

    blocker_dir = Path("/tmp/opencode/cmenpy-no-numba")
    blocker_dir.mkdir(parents=True, exist_ok=True)
    (blocker_dir / "sitecustomize.py").write_text(_BLOCKER, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(blocker_dir), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert "no-numba-ok" in result.stdout


def test_bare_precompile_casts_array_protocol_arguments() -> None:
    """A spec-less kernel still receives array-protocol objects as ndarrays."""

    @cm.precompile
    def double(X):
        return X * 2.0

    result = double(_Box(np.ones((2, 2))))

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, 2.0)


def test_bare_vectorize_casts_array_protocol_arguments() -> None:
    @cm.vectorize
    def half(x):
        return 0.5 * x

    result = half(_Box(np.array([2.0, 4.0])))

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [1.0, 2.0])


def test_python_mode_casts_array_protocol_arguments() -> None:
    @cm.precompile(numba=False)
    def double(X):
        return X * 2.0

    result = double(_Box(np.ones((2, 2))))

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, 2.0)


def test_declared_specs_still_control_the_dtype() -> None:
    @cm.precompile(cm.Array[float, 2])
    def double(X):
        return X * 2.0

    result = double(_Box(np.ones((2, 2), dtype=np.float32)))

    assert result.dtype == np.float64


def test_autocast_preserves_scalar_results() -> None:
    @cm.precompile
    def total(X):
        return float(np.sum(X))

    result = total(_Box(np.ones((2, 2))))

    assert np.ndim(result) == 0
    assert not isinstance(result, np.ndarray)
    assert result == 4.0


def test_autocast_preserves_tuple_results() -> None:
    @cm.precompile
    def pair(X):
        return X, X

    left, right = pair(_Box(np.ones((2, 2))))

    assert isinstance(left, np.ndarray)
    assert isinstance(right, np.ndarray)


def test_lists_are_not_forced_into_arrays() -> None:
    """The automatic cast never touches plain Python lists."""
    seen: list[typing.Any] = []

    @cm.precompile(numba=False)
    def record(values):
        seen.append(type(values))

        return values[0]

    assert record([2.0, 3.0]) == 2.0
    assert seen == [list]


@pytest.mark.skipif(not backend.HAS_NUMBA, reason="needs the numba backend")
def test_prepare_true_compiles_at_construction() -> None:
    outer = _template_model()
    kernel = vars(outer.model_class)["scale"]

    model = outer(seed=1, numba=True, prepare=True)

    assert kernel._cache
    assert model.prepare()["scale"] != ["<python>"]


def test_prepare_false_keeps_lazy_behaviour() -> None:
    outer = _template_model()
    kernel = vars(outer.model_class)["scale"]

    model = outer(seed=1, numba=False, prepare=False)

    assert kernel._cache == {}

    assert model.prepare()["scale"] == ["<python>"]


def test_prepare_true_without_numba_reports_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "HAS_NUMBA", False)

    model = _template_model()(seed=1, prepare=True)

    assert model.prepare()["scale"] == ["<python>"]


def test_prepare_true_demands_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "HAS_NUMBA", False)

    with pytest.raises(ImportError, match="cmenpy needs numba"):
        _template_model()(seed=1, numba=True, prepare=True)
