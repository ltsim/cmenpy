"""Tests for the `@cm.module` Operator Module and `cm.Constant` template fields."""

import typing

import numpy as np
import pytest

import cmenpy as cm
from cmenpy.compile.kernel import _Kernel


def _dispatcher(kernel: typing.Any) -> typing.Any:
    """Return the underlying numba object of a decorated kernel."""
    inner = kernel.__func__ if isinstance(kernel, staticmethod) else kernel
    numba_obj = getattr(inner, "__numba__", inner)

    return inner if isinstance(numba_obj, str) else numba_obj


def sphere(x: typing.Any) -> float:
    return float(np.sum(x**2))


def test_module_rejects_non_class() -> None:
    with pytest.raises(TypeError):
        cm.module(sphere)  # type: ignore[arg-type]


def test_module_marks_class_and_keeps_identity() -> None:
    @cm.module
    class Ops:
        pass

    assert Ops.__operator_module__ is True


def test_module_standalone_constant_is_readonly() -> None:
    @cm.module
    class Ops:
        p: cm.Constant[float, 0.05]

    assert Ops.p == 0.05

    obj = Ops()
    assert obj.p == 0.05

    with pytest.raises(AttributeError):
        obj.p = 0.1

    with pytest.raises(AttributeError):
        del obj.p


def test_constant_type_mismatch_raises_at_decoration() -> None:
    # Python 3.14 evaluates class annotations lazily (PEP 649), so the value
    # check fires when ``@cm.module`` resolves the type hints, not the body.
    with pytest.raises(TypeError):

        @cm.module
        class Bad:
            p: cm.Constant[int, 1.5]


def test_constant_requires_a_value() -> None:
    with pytest.raises(TypeError):
        cm.Constant[float]  # type: ignore[misc]


def test_constant_is_available_on_template_model() -> None:
    captured: dict[str, typing.Any] = {}

    @cm.module
    class Ops:
        scale: cm.Constant[float, 0.25]

    @cm.template
    class Model(Ops):
        a: cm.Argument[int, (1, 10), 5]

        def initialize(self, ctx) -> None:
            captured["a"] = self.a
            captured["scale"] = self.scale

        def evolve(self, e, ctx) -> None:
            pass

    Model(seed=1).solve(sphere, cm.Bounds[(-1, 1)], 1, 2)

    assert captured == {"a": 5, "scale": 0.25}


def test_constant_cannot_be_passed_to_constructor() -> None:
    @cm.module
    class Ops:
        p: cm.Constant[float, 0.1]

    @cm.template
    class Model(Ops):
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

    with pytest.raises(TypeError):
        Model(p=0.5, seed=1)


def test_positional_field_skips_constant() -> None:
    captured: dict[str, int] = {}

    @cm.module
    class Ops:
        p: cm.Constant[float, 0.05]

    @cm.template
    class Model(Ops):
        a: cm.Argument[int, (1, 10)]

        def initialize(self, ctx) -> None:
            captured["a"] = self.a

        def evolve(self, e, ctx) -> None:
            pass

    # The first positional is the seed; the next one must land on `a`, not `p`.
    Model(1, 4).solve(sphere, cm.Bounds[(-1, 1)], 1, 2)

    assert captured["a"] == 4


def test_declare_rejects_constant() -> None:
    with pytest.raises(TypeError):

        @cm.declare(p=cm.Constant[float, 0.1])
        def algo(args, epoch, ctx):
            pass


def test_inherited_kernel_reported_by_prepare() -> None:
    @cm.module
    class Ops:
        @cm.precompile
        def kernel(A: cm.Array[float, 2]):
            return A * 2.0

    @cm.template
    class Model(Ops):
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

    reported = cm.prepare(Model(seed=1))

    assert "kernel" in reported
    assert reported["kernel"]


def test_module_kernel_uses_model_compile_options() -> None:
    @cm.module
    class Ops:
        @cm.precompile
        def kernel(A: cm.Array[float, 2]):
            return A * 2.0

    kernel = Ops.__dict__["kernel"]
    assert isinstance(kernel, _Kernel)

    fast = kernel.compile({"parallel": False, "fastmath": True, "cache": False})

    assert _dispatcher(fast).targetoptions["fastmath"] is True


def test_subclass_redeclaration_overrides_constant() -> None:
    @cm.module
    class Base:
        p: cm.Constant[float, 1.0]

    @cm.module
    class Child(Base):
        p: cm.Constant[float, 2.0]

    assert Base.p == 1.0
    assert Child.p == 2.0


def test_plain_helper_is_shared_across_models() -> None:
    @cm.module
    class Ops:
        scale: cm.Constant[float, 0.1]

        def helper(self, x):
            return x + self.scale

    @cm.template
    class A(Ops):
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

    @cm.template
    class B(Ops):
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

    assert A.model_class().helper(1.0) == pytest.approx(1.1)
    assert B.model_class().helper(1.0) == pytest.approx(1.1)


def test_module_constant_is_inherited_by_template_class() -> None:
    @cm.module
    class Ops:
        K: cm.Constant[int, 3]

    @cm.template
    class Model(Ops):
        def initialize(self, ctx) -> None:
            pass

        def evolve(self, e, ctx) -> None:
            pass

    assert Model.model_class.K == 3
