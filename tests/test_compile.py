import typing

import numpy as np
import numpy.typing as npt
import pytest

import cmenpy as cm
from cmenpy.compile import _Kernel

# This module exercises numba internals (dispatchers, prange kernels,
# targetoptions); without the optional backend it is skipped. See
# tests/test_optional_numba.py for the pure-Python fallback behavior.
nb = pytest.importorskip("numba")


def _dispatcher(kernel: typing.Any) -> typing.Any:
    """Return the underlying numba object of a decorated kernel."""
    inner = kernel.__func__ if isinstance(kernel, staticmethod) else kernel
    numba_obj = getattr(inner, "__numba__", inner)

    return inner if isinstance(numba_obj, str) else numba_obj


@cm.precompile(cm.Array[float, 1], cache=True)
def _cached_square(A):
    return A**2


@cm.reduce(
    [cm.Array[float, 1], cm.Scalar[float], cm.Scalar[float], cm.Array[float, 1]],
    "(n),(),()->(n)",
    cache=True,
)
def _cached_clip(row, low, up, out):
    for i in range(row.shape[0]):
        out[i] = min(max(row[i], low), up)


def test_vectorize_bare() -> None:
    @cm.vectorize
    def double(x):
        return 2.0 * x

    assert isinstance(double, _Kernel)
    np.testing.assert_allclose(double(np.array([1.0, 2.0])), [2.0, 4.0])


def test_vectorize_with_specs() -> None:
    @cm.vectorize(cm.Array, cm.Scalar)
    def affine(r, a):
        return a * r + 1.0

    np.testing.assert_allclose(affine(np.array([1.0, 2.0]), 3.0), [4.0, 7.0])


def test_vectorize_with_numba_signature() -> None:
    @cm.vectorize("float64(float64, float64)")
    def add(x, y):
        return x + y

    np.testing.assert_allclose(add(np.array([1.0]), np.array([2.0])), [3.0])


def test_precompile_bare() -> None:
    @cm.precompile
    def square(A):
        return A**2

    assert isinstance(square, _Kernel)
    np.testing.assert_allclose(square(np.array([2.0, 3.0])), [4.0, 9.0])


def test_parallel_prange() -> None:
    @cm.precompile(cm.Array, cm.Scalar, parallel=True)
    def scale_rows(A, factor):
        out = np.empty_like(A)

        for i in nb.prange(A.shape[0]):
            out[i] = A[i] * factor

        return out

    A = np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(scale_rows(A, 2.0), A * 2.0)


def test_scalar_casting() -> None:
    @cm.precompile(cm.Scalar)
    def inc(x):
        return x + 1.0

    assert inc(1) == 2.0


def test_list_is_cast_to_array() -> None:
    @cm.precompile(cm.Array)
    def total(X):
        acc = 0.0

        for i in range(X.shape[0]):
            acc += X[i]

        return acc

    assert total([1.0, 2.0, 3.0]) == 6.0


def test_template_casts_proxy_internally(monkeypatch: "pytest.MonkeyPatch") -> None:
    from cmenpy.population.attributes.base import BufferMatrixSolutions

    seen: dict[str, type] = {}
    triggered: list[int] = []

    def _spy(self, dtype=None, copy=None):
        triggered.append(1)
        return self.to_numpy()

    # A spec'd kernel must unwrap the view itself, not via ``np.asarray``.
    monkeypatch.setattr(BufferMatrixSolutions, "__array__", _spy, raising=False)

    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
            pop = cm.PopulationSwarm(buff)

            pop.compute << rng.uniform(-1, 1, (pop.size, bounds.ndim))

            out = self.halve(pop.X, 0.5)  # view -> unwrapped by the kernel
            seen["type"] = type(out)

        def evolve(self, e, ctx) -> None:
            pass

        @cm.precompile(cm.Array, cm.Scalar, parallel=True)
        def halve(X, factor):
            out = np.empty_like(X)

            for i in nb.prange(X.shape[0]):
                out[i] = X[i] * factor

            return out

    def sphere(x):
        return float(np.sum(x**2))

    model = Model(seed=1)
    model.solve(sphere, cm.Bounds[(-1, 1), (-1, 1)], 1, 3, (1, 4))

    assert seen["type"] is np.ndarray
    assert triggered == []


def test_eager_precompile_compiles_at_definition() -> None:
    @cm.precompile(cm.Array[float, 2])
    def square(A):
        return A**2

    # No call has been made, yet the kernel is already compiled.
    assert square.__func__.__numba__.signatures


def test_eager_parallel_compiles_at_definition() -> None:
    @cm.precompile(cm.Array[float, 2], cm.Scalar[float], parallel=True)
    def scale(X, factor):
        out = np.empty_like(X)

        for i in nb.prange(X.shape[0]):
            out[i] = X[i] * factor

        return out

    assert scale.__func__.__numba__.signatures


def test_eager_vectorize_compiles_at_definition() -> None:
    @cm.vectorize(cm.Scalar[float, [cm.Scalar[float]]])
    def double(x):
        return 2.0 * x

    assert double.__func__.__numba__.types


def test_typed_array_coerces_dtype() -> None:
    @cm.precompile(cm.Array[float, 2])
    def total(A):
        acc = 0.0

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                acc += A[i, j]

        return acc

    f32 = np.ones((2, 2), dtype=np.float32)

    assert total(f32) == 4.0
    # Only the precompiled float64 signature exists (input was upcast).
    assert len(total.__func__.__numba__.signatures) == 1


def test_typed_array_coerces_layout() -> None:
    @cm.precompile(cm.Array[float, 2])
    def first(A):
        return A[0, 0]

    f_order = np.asfortranarray(np.arange(4.0).reshape(2, 2))

    assert first(f_order) == 0.0
    assert len(first.__func__.__numba__.signatures) == 1


def test_cache_option() -> None:
    np.testing.assert_allclose(_cached_square(np.array([2.0, 3.0])), [4.0, 9.0])


def test_prepare_reports_kernels() -> None:
    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

        @cm.precompile(cm.Array[float, 2], cm.Scalar[float], parallel=True)
        def scale(X, factor):
            out = np.empty_like(X)

            for i in nb.prange(X.shape[0]):
                out[i] = X[i] * factor

            return out

    model = Model(seed=1)
    reported = cm.prepare(model)

    assert "scale" in reported
    assert reported["scale"]


def test_bare_kernels_round_trip_buffer_views() -> None:
    """Spec-less kernels accept ``pop.X`` directly and return plain ndarrays."""
    seen: list[typing.Any] = []

    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pop = cm.PopulationSwarm(ctx.buff)
            pop.compute << np.ones((pop.size, ctx.bounds.ndim))

        def evolve(self, e, ctx) -> None:
            pop = cm.PopulationSwarm(ctx.buff)
            result = self.double(pop.X)
            seen.append(result)
            pop.X << result

        @cm.precompile
        def double(X):
            return X * 2.0

    Model(seed=1).solve(
        lambda x: float(np.sum(x**2)), cm.Bounds[(-1, 1), (-1, 1)], 1, 2
    )

    assert seen
    assert all(isinstance(result, np.ndarray) for result in seen)
    np.testing.assert_allclose(seen[-1], 2.0)


def test_reduce_compiles_at_definition() -> None:
    @cm.reduce(
        [
            cm.Array[float, 1],
            cm.Scalar[float],
            cm.Scalar[float],
            cm.Array[float, 1],
        ],
        "(n),(),()->(n)",
    )
    def clip(row, low, up, out):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)

    assert clip.__func__.__numba__.types


def test_reduce_clips_each_row() -> None:
    @cm.reduce(
        [
            cm.Array[float, 1],
            cm.Scalar[float],
            cm.Scalar[float],
            cm.Array[float, 1],
        ],
        "(n),(),()->(n)",
    )
    def clip(row, low, up, out):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)

    matrix = np.array([[-5.0, 0.5, 5.0], [10.0, -10.0, 1.0]])

    np.testing.assert_allclose(clip(matrix, -1.0, 1.0), np.clip(matrix, -1.0, 1.0))


def test_reduce_casts_dtype() -> None:
    @cm.reduce(
        [cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]],
        "(n),()->(n)",
    )
    def scale(row, factor, out):
        for i in range(row.shape[0]):
            out[i] = row[i] * factor

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    assert scale(matrix, 2.0).dtype == np.float64


def test_reduce_multiple_signatures() -> None:
    @cm.reduce(
        [
            [cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]],
            [cm.Array[np.float32, 1], cm.Scalar[float], cm.Array[np.float32, 1]],
        ],
        "(n),()->(n)",
    )
    def scale(row, factor, out):
        for i in range(row.shape[0]):
            out[i] = row[i] * factor

    assert len(scale.__func__.__numba__.types) == 2

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    np.testing.assert_allclose(scale(matrix, 2.0), matrix * 2.0)


def test_reduce_prepare_and_cache() -> None:
    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

        @cm.reduce(
            [cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]],
            "(n),()->(n)",
        )
        def scale(row, factor, out):
            for i in range(row.shape[0]):
                out[i] = row[i] * factor

    model = Model(seed=1)
    reported = cm.prepare(model)

    assert "scale" in reported
    assert reported["scale"]

    np.testing.assert_allclose(
        _cached_clip(np.array([[-5.0, 0.0, 5.0]]), -1.0, 1.0), [[-1.0, 0.0, 1.0]]
    )


def test_precompile_annotated_eager() -> None:
    @cm.precompile
    def square(A: "Array[float, 2]"):
        return A**2

    # No call has been made, yet the annotated kernel is already compiled.
    assert square.__func__.__numba__.signatures

    f32 = np.ones((2, 2), dtype=np.float32)
    np.testing.assert_allclose(square(f32), np.ones((2, 2)))


def test_precompile_annotated_lazy() -> None:
    @cm.precompile
    def total(X: "Array"):
        acc = 0.0

        for i in range(X.shape[0]):
            acc += X[i]

        return acc

    assert not total.__func__.__numba__.signatures
    assert total([1.0, 2.0, 3.0]) == 6.0
    assert total.__func__.__numba__.signatures


def test_parallel_annotated() -> None:
    @cm.precompile(parallel=True)
    def scale_rows(A: "Array[float, 2]", factor: "Scalar[float]"):
        out = np.empty_like(A)

        for i in nb.prange(A.shape[0]):
            out[i] = A[i] * factor

        return out

    assert scale_rows.__func__.__numba__.signatures

    A = np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(scale_rows(A, 2.0), A * 2.0)


def test_vectorize_annotated_eager() -> None:
    @cm.vectorize
    def affine(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]":
        return a * r + 1.0

    assert affine.__func__.__numba__.types
    np.testing.assert_allclose(affine(np.array([1.0, 2.0]), 3.0), [4.0, 7.0])


def test_vectorize_annotated_lazy() -> None:
    @cm.vectorize
    def affine(r: "Array", a: "Scalar"):
        return a * r + 1.0

    np.testing.assert_allclose(affine(np.array([1.0, 2.0]), 3.0), [4.0, 7.0])


def test_vectorize_annotated_partial_raises() -> None:
    with pytest.raises(TypeError):

        @cm.vectorize
        def bad(x: "Scalar[float]"):
            return x


def test_reduce_annotated_layout_keyword() -> None:
    @cm.reduce(layout="(n),()->(n)")
    def scale(row: "Array[float, 1]", factor: "Scalar[float]", out: "Array[float, 1]"):
        for i in range(row.shape[0]):
            out[i] = row[i] * factor

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    assert scale(matrix, 2.0).dtype == np.float64


def test_numba_signature_option_wins_over_annotations() -> None:
    @cm.vectorize("float64(float64, float64)")
    def add(x: "Scalar[float]", y: "Scalar[float]") -> "Scalar[float]":
        return x + y

    np.testing.assert_allclose(add(np.array([1.0]), np.array([2.0])), [3.0])


def test_reduce_annotated() -> None:
    @cm.reduce("(n),(),()->(n)")
    def clip(
        row: "Array[float, 1]",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 1]",
    ):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)

    assert clip.__func__.__numba__.types

    matrix = np.array([[-5.0, 0.5, 5.0], [10.0, -10.0, 1.0]])
    np.testing.assert_allclose(clip(matrix, -1.0, 1.0), np.clip(matrix, -1.0, 1.0))


def test_annotated_metadata_is_used() -> None:
    @cm.precompile
    def square(A: typing.Annotated[npt.NDArray[np.float64], cm.Array[float, 2]]):
        return A**2

    assert square.__func__.__numba__.signatures
    np.testing.assert_allclose(square(np.ones((2, 2))), np.ones((2, 2)))


def test_annotated_string_metadata_is_used() -> None:
    @cm.precompile
    def square(A: typing.Annotated[np.ndarray, "Array[float, 2]"]):
        return A**2

    assert square.__func__.__numba__.signatures
    np.testing.assert_allclose(square(np.ones((2, 2))), np.ones((2, 2)))


def test_annotated_ignores_plain_hints() -> None:
    @cm.precompile
    def norm(X: npt.NDArray[np.float64]) -> float:
        return float(np.sum(X))

    # A plain type hint is not a spec, so the kernel stays lazy.
    assert not _dispatcher(norm).signatures
    assert norm(np.array([1.0, 2.0])) == 3.0


def test_positional_specs_take_precedence() -> None:
    @cm.precompile(cm.Array[np.float32, 2])
    def square(A: "Array[float, 2]"):
        return A**2

    # The positional float32 spec wins over the float annotation.
    assert "float32" in str(square.__func__.__numba__.signatures[0])


def test_annotated_with_future_annotations() -> None:
    namespace: dict[str, typing.Any] = {}

    exec(
        "from __future__ import annotations\n"
        "import cmenpy as cm\n"
        "def square(A: cm.Array[float, 2]):\n"
        "    return A ** 2\n",
        namespace,
    )

    square = cm.precompile(namespace["square"])

    assert square.__func__.__numba__.signatures
    np.testing.assert_allclose(square(np.ones((2, 2))), np.ones((2, 2)))


def test_array_dimension_labels() -> None:
    spec = cm.Array[float, "m,n"]

    assert spec.dims == ("m", "n")
    assert spec.rank == 2
    assert spec.numba_type() == nb.types.Array(nb.float64, 2, "C")


def test_contract_layouts() -> None:
    assert (
        cm.Contract[[float, float, ..., ..., ...], float].layout
        == "(n),(n),(),(),()->(n)"
    )
    assert cm.Contract[[float, cm.Array[float]], float].layout == "(n),(k0)->(n)"
    assert (
        cm.Contract[
            [cm.Array[float, "m,n"], cm.Array[float, "n,k"]], cm.Array[float, "m,k"]
        ].layout
        == "(m,n),(n,k)->(m,k)"
    )
    assert (
        cm.Contract[[float, cm.Array[float, "m"]], cm.Array[float, "n,m"]].layout
        == "(n),(m)->(n,m)"
    )


def test_contract_scalars_and_vectors() -> None:
    @cm.reduce(cm.Contract[[float, float, ..., ..., ...], float])
    def combine(a, b, s1, s2, s3, out):
        for i in range(a.shape[0]):
            out[i] = a[i] + b[i] * (s1 + s2 + s3)

    A = np.arange(3.0)
    np.testing.assert_allclose(combine(A, np.ones(3), 1.0, 2.0, 3.0), A + 6.0)


def test_contract_private_dim() -> None:
    @cm.reduce(cm.Contract[[float, cm.Array[float]], float])
    def scale(x, w, out):
        for i in range(x.shape[0]):
            out[i] = x[i] * w[i]

    x = np.array([1.0, 2.0])
    w = np.array([3.0, 4.0])

    np.testing.assert_allclose(scale(x, w), [3.0, 8.0])


def test_contract_matmul() -> None:
    @cm.reduce(
        cm.Contract[
            [cm.Array[float, "m,n"], cm.Array[float, "n,k"]], cm.Array[float, "m,k"]
        ]
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


def test_contract_outer() -> None:
    @cm.reduce(cm.Contract[[float, cm.Array[float, "m"]], cm.Array[float, "n,m"]])
    def outer(x, y, out):
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                out[i, j] = x[i] * y[j]

    x = np.arange(3.0)
    y = np.arange(4.0)

    np.testing.assert_allclose(outer(x, y), np.outer(x, y))


def test_contract_batch_matmul() -> None:
    @cm.reduce(
        cm.Contract[
            [cm.Array[float, "b,m,n"], cm.Array[float, "b,n,k"]],
            cm.Array[float, "b,m,k"],
        ]
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


def test_contract_from_annotations() -> None:
    @cm.reduce
    def matmul(
        a: "Array[float, 'm,n']",
        b: "Array[float, 'n,k']",
        out: "Array[float, 'm,k']",
    ):
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                acc = 0.0

                for k in range(a.shape[1]):
                    acc += a[i, k] * b[k, j]

                out[i, j] = acc

    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)

    np.testing.assert_allclose(matmul(A, B), A @ B)


def test_contract_missing_output_dim_raises() -> None:
    with pytest.raises(TypeError):
        cm.Contract[[cm.Array[float, "n"]], cm.Array[float, "m"]]


def test_contract_scalar_output_raises() -> None:
    with pytest.raises(TypeError):
        cm.Contract[[float], ...]


def test_contract_annotations_require_all_params() -> None:
    with pytest.raises(TypeError):

        @cm.reduce
        def bad(a: "Array[float, 'n']", out): ...


def test_jit_alias_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning):

        @cm.jit
        def square(A):
            return A**2

    assert isinstance(square, _Kernel)
    np.testing.assert_allclose(square(np.array([2.0, 3.0])), [4.0, 9.0])


def test_precompile_matches_jit_alias() -> None:
    with pytest.warns(DeprecationWarning):

        @cm.jit(cm.Array[float, 2])
        def via_alias(A):
            return A**2

    @cm.precompile(cm.Array[float, 2])
    def direct(A):
        return A**2

    A = np.arange(4.0).reshape(2, 2)
    np.testing.assert_allclose(via_alias(A), direct(A))


def test_precompile_fastmath_option() -> None:
    @cm.precompile(fastmath=True)
    def square(A: "Array[float, 1]"):
        return A**2

    assert square.__func__.__numba__.targetoptions["fastmath"] is True
    np.testing.assert_allclose(square(np.array([2.0, 3.0])), [4.0, 9.0])


def test_precompile_parallel_option() -> None:
    @cm.precompile(parallel=True)
    def scale_rows(A: "Array[float, 2]"):
        out = np.empty_like(A)

        for i in nb.prange(A.shape[0]):
            out[i] = A[i] * 2.0

        return out

    assert scale_rows.__func__.__numba__.targetoptions["parallel"] is True

    A = np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(scale_rows(A), A * 2.0)


def test_precompile_fastmath_and_parallel() -> None:
    @cm.precompile(fastmath=True, parallel=True)
    def bump(A: "Array[float, 2]"):
        out = np.empty_like(A)

        for i in nb.prange(A.shape[0]):
            out[i] = A[i] + 1.0

        return out

    options = bump.__func__.__numba__.targetoptions

    assert options["fastmath"] is True
    assert options["parallel"] is True


def test_model_compile_options_are_captured() -> None:
    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

        @cm.precompile(parallel=True)
        def kernel(A: "Array[float, 2]"):
            return A * 2.0

    model = Model(seed=1, parallel=True, fastmath=True, cache=True)

    assert model.compile_options == {
        "parallel": True,
        "fastmath": True,
        "cache": True,
    }


def test_model_parallel_enables_eligible_precompile() -> None:
    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

        @cm.precompile(parallel=True)
        def kernel(A: "Array[float, 2]"):
            out = np.empty_like(A)

            for i in nb.prange(A.shape[0]):
                out[i] = A[i] * 2.0

            return out

    kernel = vars(Model.model_class)["kernel"]

    off = kernel.compile({"parallel": False, "fastmath": False, "cache": False})
    on = kernel.compile({"parallel": True, "fastmath": False, "cache": False})

    assert _dispatcher(off).targetoptions.get("parallel", False) is False
    assert _dispatcher(on).targetoptions["parallel"] is True


def test_model_parallel_enables_ufunc_reduce() -> None:
    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

        @cm.reduce
        def kernel(a: "Array[float, 'n']", out: "Array[float, 'n']"):
            for i in range(a.shape[0]):
                out[i] = a[i] * 2.0

    kernel = vars(Model.model_class)["kernel"]
    on = kernel.compile({"parallel": True, "fastmath": False, "cache": False})

    assert isinstance(_dispatcher(on), np.ufunc)


def test_model_fastmath_reaches_kernels() -> None:
    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

        @cm.precompile
        def kernel(A: "Array[float, 2]"):
            return A * A

    kernel = vars(Model.model_class)["kernel"]
    fast = kernel.compile({"parallel": False, "fastmath": True, "cache": False})

    assert _dispatcher(fast).targetoptions["fastmath"] is True


def test_jit_alias_accepts_options() -> None:
    with pytest.warns(DeprecationWarning):

        @cm.jit(fastmath=True)
        def square(A: "Array[float, 1]"):
            return A**2

    assert square.__func__.__numba__.targetoptions["fastmath"] is True


def test_vectorize_fastmath_option() -> None:
    @cm.vectorize(fastmath=True)
    def affine(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]":
        return a * r + 1.0

    assert affine.__func__.__numba__.targetoptions["fastmath"] is True
    np.testing.assert_allclose(affine(np.array([1.0, 2.0]), 3.0), [4.0, 7.0])


def test_vectorize_parallel_option() -> None:
    @cm.vectorize(parallel=True)
    def affine(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]":
        return a * r + 1.0

    # ``target="parallel"`` compiles down to a numpy ufunc.
    assert isinstance(affine.__func__.__numba__, np.ufunc)
    np.testing.assert_allclose(affine(np.array([1.0, 2.0]), 3.0), [4.0, 7.0])


def test_reduce_fastmath_option() -> None:
    @cm.reduce(fastmath=True)
    def add(a: "Array[float, 'n']", b: "Array[float, 'n']", out: "Array[float, 'n']"):
        for i in range(a.shape[0]):
            out[i] = a[i] + b[i]

    np.testing.assert_allclose(
        add(np.array([1.0, 2.0]), np.array([3.0, 4.0])), [4.0, 6.0]
    )


def test_reduce_parallel_option() -> None:
    @cm.reduce(parallel=True)
    def add(a: "Array[float, 'n']", b: "Array[float, 'n']", out: "Array[float, 'n']"):
        for i in range(a.shape[0]):
            out[i] = a[i] + b[i]

    # ``target="parallel"`` compiles down to a numpy ufunc.
    assert isinstance(add.__func__.__numba__, np.ufunc)
    np.testing.assert_allclose(
        add(np.array([1.0, 2.0]), np.array([3.0, 4.0])), [4.0, 6.0]
    )
