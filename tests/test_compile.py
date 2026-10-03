import numba as nb
import numpy as np

import cmenpy as cm


@cm.jit(cm.Array[float, 1], cache=True)
def _cached_square(A):
    return A**2


@cm.guvectorize(
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

    assert isinstance(double, staticmethod)
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


def test_jit_bare() -> None:
    @cm.jit
    def square(A):
        return A**2

    assert isinstance(square, staticmethod)
    np.testing.assert_allclose(square(np.array([2.0, 3.0])), [4.0, 9.0])


def test_parallel_prange() -> None:
    @cm.parallel(cm.Array, cm.Scalar)
    def scale_rows(A, factor):
        out = np.empty_like(A)

        for i in nb.prange(A.shape[0]):
            out[i] = A[i] * factor

        return out

    A = np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(scale_rows(A, 2.0), A * 2.0)


def test_scalar_casting() -> None:
    @cm.jit(cm.Scalar)
    def inc(x):
        return x + 1.0

    assert inc(1) == 2.0


def test_list_is_cast_to_array() -> None:
    @cm.jit(cm.Array)
    def total(X):
        acc = 0.0

        for i in range(X.shape[0]):
            acc += X[i]

        return acc

    assert total([1.0, 2.0, 3.0]) == 6.0


def test_template_autocasts_proxy() -> None:
    seen: dict[str, type] = {}

    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
            pop = cm.PopulationSwarm(buff)

            pop.compute << rng.uniform(-1, 1, (pop.size, bounds.ndim))

            out = self.halve(pop.X, 0.5)  # BufferMatrixSolutions proxy -> ndarray
            seen["type"] = type(out)

        def evolve(self, e, ctx) -> None:
            pass

        @cm.parallel(cm.Array, cm.Scalar)
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


def test_eager_jit_compiles_at_definition() -> None:
    @cm.jit(cm.Array[float, 2])
    def square(A):
        return A**2

    # No call has been made, yet the kernel is already compiled.
    assert square.__func__.__numba__.signatures


def test_eager_parallel_compiles_at_definition() -> None:
    @cm.parallel(cm.Array[float, 2], cm.Scalar[float])
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
    @cm.jit(cm.Array[float, 2])
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
    @cm.jit(cm.Array[float, 2])
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

        @cm.parallel(cm.Array[float, 2], cm.Scalar[float])
        def scale(X, factor):
            out = np.empty_like(X)

            for i in nb.prange(X.shape[0]):
                out[i] = X[i] * factor

            return out

    model = Model(seed=1)
    reported = cm.prepare(model)

    assert "scale" in reported
    assert reported["scale"]


def test_guvectorize_compiles_at_definition() -> None:
    @cm.guvectorize(
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


def test_guvectorize_clips_each_row() -> None:
    @cm.guvectorize(
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


def test_guvectorize_casts_dtype() -> None:
    @cm.guvectorize(
        [cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]],
        "(n),()->(n)",
    )
    def scale(row, factor, out):
        for i in range(row.shape[0]):
            out[i] = row[i] * factor

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    assert scale(matrix, 2.0).dtype == np.float64


def test_guvectorize_multiple_signatures() -> None:
    @cm.guvectorize(
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


def test_guvectorize_prepare_and_cache() -> None:
    @cm.template
    class Model:
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

        @cm.guvectorize(
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


