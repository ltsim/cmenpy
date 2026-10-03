import functools
import inspect
import typing

import numba as nb
import numpy as np

P = typing.ParamSpec("P")
R = typing.TypeVar("R")

_Caster = typing.Callable[[typing.Any], typing.Any]


def _dtype_name(dtype: typing.Any) -> str:
    return str(np.dtype(dtype).name)


def _elementwise_caster(dtype: typing.Any) -> _Caster:
    """Cast arrays to ``dtype`` and scalars to a ``dtype`` scalar."""
    if dtype is None:
        return np.asarray

    _dtype = np.dtype(dtype)

    def cast(value: typing.Any) -> typing.Any:
        if np.ndim(value) == 0:
            return _dtype.type(value)

        return np.asarray(value, dtype=_dtype)

    return cast


class _ArraySpec:
    """Marker describing how an array parameter is cast and typed."""

    __slots__ = ("dtype", "layout", "ndim")

    def __init__(
        self,
        dtype: typing.Any = None,
        ndim: int | None = None,
        layout: str = "C",
    ) -> None:
        self.dtype = dtype
        self.ndim = ndim
        self.layout = layout

    def __getitem__(self, params: typing.Any) -> "_ArraySpec":
        items = params if isinstance(params, tuple) else (params,)
        dtype = items[0]
        ndim = items[1] if len(items) > 1 else None
        layout = items[2] if len(items) > 2 else "C"

        return _ArraySpec(dtype, ndim, layout)

    @property
    def typed(self) -> bool:
        return self.dtype is not None and self.ndim is not None

    def numba_type(self) -> typing.Any:
        return nb.types.Array(  # type: ignore[no-untyped-call]
            nb.from_dtype(np.dtype(self.dtype)), self.ndim, self.layout
        )

    def caster(self) -> _Caster:
        if self.dtype is None:
            return np.asarray

        _dtype = np.dtype(self.dtype)

        if self.layout == "F":
            return lambda value: np.asfortranarray(value, dtype=_dtype)

        if self.layout == "A":
            return lambda value: np.asarray(value, dtype=_dtype)

        return lambda value: np.ascontiguousarray(value, dtype=_dtype)

    def __repr__(self) -> str:
        return f"cm.Array[{self.dtype!r}, {self.ndim!r}, {self.layout!r}]"


class _ScalarSpec:
    """Marker describing how a scalar parameter is cast and typed."""

    __slots__ = ("dtype",)

    def __init__(self, dtype: typing.Any = None) -> None:
        self.dtype = dtype

    def __getitem__(self, params: typing.Any) -> "_ScalarSpec | _Signature":
        items = params if isinstance(params, tuple) else (params,)

        if len(items) == 1:
            return _ScalarSpec(items[0])

        result = _ScalarSpec(items[0])
        args = tuple(items[1])

        return _Signature(result, args)

    @property
    def typed(self) -> bool:
        return self.dtype is not None

    def numba_type(self) -> typing.Any:
        return nb.from_dtype(np.dtype(self.dtype))

    def caster(self) -> _Caster:
        if self.dtype is None:
            return float

        _dtype = np.dtype(self.dtype)

        return lambda value: _dtype.type(value)

    def __repr__(self) -> str:
        return f"cm.Scalar[{self.dtype!r}]"


class _Signature:
    """Full eager signature for ``@cm.vectorize`` (result + argument specs)."""

    __slots__ = ("args", "result")

    def __init__(self, result: _ScalarSpec, args: tuple[_ScalarSpec, ...]) -> None:
        self.result = result
        self.args = args

    def numba_signature(self) -> str:
        result = _dtype_name(self.result.dtype)
        args = ", ".join(_dtype_name(arg.dtype) for arg in self.args)

        return f"{result}({args})"

    def __repr__(self) -> str:
        return f"cm.Scalar[{self.result.dtype!r}, {list(self.args)!r}]"


#: Marker: receive this parameter as an ndarray (optionally typed).
Array = _ArraySpec()

#: Marker: receive this parameter as a scalar (optionally typed).
Scalar = _ScalarSpec()

_SPECS = (_ArraySpec, _ScalarSpec, _Signature)


def _split(
    args: tuple[typing.Any, ...],
) -> tuple[tuple[typing.Any, ...], tuple[typing.Any, ...]]:
    """Separate leading cast markers from the numba options."""
    if args and all(isinstance(a, _SPECS) for a in args):
        return args, ()

    return (), args


def _resolve(
    specs: tuple[typing.Any, ...],
) -> tuple[tuple[typing.Any, ...], list[_Caster]]:
    """Turn cast markers into numba arguments and per-parameter casters."""
    if not specs:
        return (), []

    if len(specs) == 1 and isinstance(specs[0], _Signature):
        signature = specs[0]
        casters = [_elementwise_caster(arg.dtype) for arg in signature.args]

        return (signature.numba_signature(),), casters

    casters = [spec.caster() for spec in specs]

    if all(spec.typed for spec in specs):
        types = tuple(spec.numba_type() for spec in specs)

        return (types,), casters

    return (), casters


def _autocast(
    compiled: typing.Any,
    casters: list[_Caster],
    func: typing.Callable[..., typing.Any],
) -> typing.Any:
    """Wrap ``compiled`` so its arguments are cast before being passed."""
    if not casters:
        return compiled

    names = list(inspect.signature(func).parameters)
    mapping = {
        name: caster
        for name, caster in zip(names, casters)
        if caster is not None
    }

    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        casted = tuple(
            mapping[names[i]](value)
            if i < len(names) and names[i] in mapping
            else value
            for i, value in enumerate(args)
        )
        casted_kwargs = {
            name: mapping[name](value) if name in mapping else value
            for name, value in kwargs.items()
        }

        return compiled(*casted, **casted_kwargs)

    wrapper.__numba__ = compiled  # type: ignore[attr-defined]

    return wrapper


def _template(
    compiler: typing.Callable[..., typing.Any],
    **defaults: typing.Any,
) -> typing.Callable[..., typing.Any]:
    """Build a decorator for ``compiler`` supporting bare, option and spec calls."""

    def decorator(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        specs, options = _split(args)
        options_map = {**defaults, **kwargs}

        if len(options) == 1 and not kwargs and callable(options[0]):
            return staticmethod(compiler(options[0], **options_map))

        numba_args, casters = _resolve(specs)

        def wrap(func: typing.Callable[..., typing.Any]) -> typing.Any:
            compiled = compiler(*numba_args, *options, **options_map)(func)

            return staticmethod(_autocast(compiled, casters, func))

        return wrap

    return decorator


def vectorize(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Vectorize a static method, optionally precompiling it.

    Precompilation needs a full signature because numba requires the return
    type, e.g. ``cm.vectorize(cm.Scalar[float, [cm.Scalar[float], cm.Scalar[float]]])``.
    """
    if args and all(isinstance(a, _SPECS) for a in args):
        is_full = len(args) == 1 and isinstance(args[0], _Signature)

        if not is_full and any(getattr(a, "typed", False) for a in args):
            raise TypeError(
                "Precompiled @cm.vectorize needs a full signature, e.g. "
                "cm.vectorize(cm.Scalar[float, [cm.Scalar[float]]])."
            )

    return _template(nb.vectorize)(*args, **kwargs)


def jit(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Compile a static method with :func:`numba.njit`."""
    return _template(nb.njit)(*args, **kwargs)


def parallel(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Compile a static method with :func:`numba.njit` and ``parallel=True``."""
    return _template(nb.njit, parallel=True)(*args, **kwargs)


def _parse_signatures(
    signatures: typing.Any,
) -> list[tuple[typing.Any, ...]]:
    """Normalize a flat list of specs into a list of core signatures."""
    if isinstance(signatures, _SPECS):
        signatures = [signatures]

    items = list(signatures)

    if not items:
        raise TypeError("cm.guvectorize requires at least one signature.")

    if all(isinstance(item, _SPECS) for item in items):
        return [tuple(items)]

    return [tuple(signature) for signature in items]


def _count_inputs(layout: str) -> int:
    left = layout.split("->")[0].strip()

    return len(left.split(",")) if left else 0


def guvectorize(
    signatures: typing.Any,
    layout: str,
    **kwargs: typing.Any,
) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Any]:
    """Build a generalized ufunc (``numba.guvectorize``) static method.

    ``signatures`` follows numba's style: a list of core type specs, where a
    flat list means one signature and a nested list means several, e.g.
    ``[cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]]`` with the
    layout ``"(n),()->(n)"``. Inputs are cast from the first signature; outputs
    (after ``->``) are allocated by numba.
    """
    parsed = _parse_signatures(signatures)

    for signature in parsed:
        for spec in signature:
            if not getattr(spec, "typed", False):
                raise TypeError(
                    "cm.guvectorize requires fully typed specs, e.g. "
                    "cm.Array[float, 1] or cm.Scalar[float]."
                )

    ftylist = [tuple(spec.numba_type() for spec in sig) for sig in parsed]
    n_inputs = _count_inputs(layout)
    first = parsed[0]

    if n_inputs > len(first):
        raise TypeError(
            f"Layout '{layout}' expects {n_inputs} inputs, "
            f"but {len(first)} specs were given."
        )

    casters = [first[index].caster() for index in range(n_inputs)]

    def wrap(func: typing.Callable[..., typing.Any]) -> typing.Any:
        compiled = nb.guvectorize(ftylist, layout, **kwargs)(  # type: ignore[no-untyped-call]
            func
        )

        return staticmethod(_autocast(compiled, casters, func))

    return wrap


def kernels(cls: type[typing.Any]) -> dict[str, list[str]]:
    """Return the precompiled numba kernels declared on ``cls``."""
    found: dict[str, list[str]] = {}

    for name, attr in vars(cls).items():
        func = attr.__func__ if isinstance(attr, staticmethod) else attr
        numba_obj = getattr(func, "__numba__", func)
        signatures = getattr(numba_obj, "signatures", None)

        if signatures is None:
            signatures = getattr(numba_obj, "types", None)

        if signatures is None:
            continue

        found[name] = [str(signature) for signature in signatures]

    return found


def prepare(target: typing.Any) -> dict[str, list[str]]:
    """Force/report the precompilation of the kernels of a model or class."""
    runner = getattr(target, "prepare", None)

    if callable(runner):
        return typing.cast(dict[str, list[str]], runner())

    if isinstance(target, type):
        return kernels(target)

    raise TypeError(f"Cannot prepare '{target}'.")


__all__ = [
    "Array",
    "Scalar",
    "guvectorize",
    "jit",
    "kernels",
    "parallel",
    "prepare",
    "vectorize",
]
