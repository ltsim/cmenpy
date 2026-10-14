import functools
import inspect
import itertools
import typing
import warnings

import numba as nb
import numpy as np

P = typing.ParamSpec("P")
R = typing.TypeVar("R")

_Caster = typing.Callable[[typing.Any], typing.Any]
_Spec = typing.Any  # an _ArraySpec, _ScalarSpec or _Signature marker


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
    """Marker describing how an array parameter is cast and typed.

    ``ndim`` can be an integer (the rank) or a string of comma-separated
    dimension labels (e.g. ``"m,n"``) used by generalized ufuncs.
    """

    __slots__ = ("dtype", "layout", "ndim")

    def __init__(
        self,
        dtype: typing.Any = None,
        ndim: int | str | None = None,
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
    def dims(self) -> tuple[str, ...] | None:
        """Dimension labels when ``ndim`` is a string, else ``None``."""
        if isinstance(self.ndim, str):
            return tuple(part.strip() for part in self.ndim.split(",") if part.strip())

        return None

    @property
    def rank(self) -> int | None:
        """Number of dimensions, whether ``ndim`` is a rank or labels."""
        if isinstance(self.ndim, str):
            dims = self.dims

            return len(dims) if dims else None

        return self.ndim

    @property
    def typed(self) -> bool:
        return self.dtype is not None and self.rank is not None

    def numba_type(self) -> typing.Any:
        return nb.types.Array(  # type: ignore[no-untyped-call]
            nb.from_dtype(np.dtype(self.dtype)), self.rank, self.layout
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

    def __init__(
        self,
        result: "_ArraySpec | _ScalarSpec",
        args: "tuple[_ArraySpec | _ScalarSpec, ...]",
    ) -> None:
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


def _annotation_namespace(
    func: typing.Callable[..., typing.Any],
) -> dict[str, typing.Any]:
    """Namespace used to resolve string annotations such as ``"Array[float, 2]"``."""
    return {
        **getattr(func, "__globals__", {}),
        "Array": Array,
        "Scalar": Scalar,
    }


def _to_spec(
    value: typing.Any,
    namespace: dict[str, typing.Any],
    depth: int = 0,
) -> "_ArraySpec | _ScalarSpec | None":
    """Normalize a single annotation into a cast spec, or ``None`` if it is not one.

    Accepts an already-built marker, a string such as ``"Array[float, 2]"`` and a
    ``typing.Annotated[...]`` whose metadata carries the marker. Anything else
    (plain type hints, ``None``, ...) is ignored.
    """
    if isinstance(value, (_ArraySpec, _ScalarSpec)):
        return value

    if isinstance(value, str):
        if depth >= 4:
            return None

        try:
            resolved = eval(value, dict(namespace))
        # An arbitrary annotation expression may raise anything; ignore non-specs.
        except Exception:  # noqa: BLE001
            return None

        return _to_spec(resolved, namespace, depth + 1)

    if typing.get_origin(value) is typing.Annotated:
        for metadata in typing.get_args(value)[1:]:
            spec = _to_spec(metadata, namespace, depth + 1)

            if spec is not None:
                return spec

    return None


def _function_hints(
    func: typing.Callable[..., typing.Any],
) -> dict[str, typing.Any]:
    """Resolve the annotations of ``func``, keeping ``Annotated`` metadata."""
    try:
        return typing.get_type_hints(func, include_extras=True)
    # Unresolvable plain-string specs (e.g. ``"Array[float, 2]"``) fail here.
    except Exception:  # noqa: BLE001
        return dict(getattr(func, "__annotations__", {}))


def _annotation_specs(
    func: typing.Callable[..., typing.Any],
) -> "tuple[_ArraySpec | _ScalarSpec | None, ...]":
    """Read the per-parameter cast specs from the annotations of ``func``."""
    hints = _function_hints(func)
    namespace = _annotation_namespace(func)
    specs: list[_ArraySpec | _ScalarSpec | None] = []

    for parameter in inspect.signature(func).parameters.values():
        annotation = hints.get(parameter.name, parameter.annotation)
        specs.append(_to_spec(annotation, namespace))

    return tuple(specs)


def _return_spec(
    func: typing.Callable[..., typing.Any],
) -> "_ArraySpec | _ScalarSpec | None":
    """Read the cast spec from the return annotation of ``func``, if any."""
    annotation = _function_hints(func).get("return")

    if annotation is None:
        return None

    return _to_spec(annotation, _annotation_namespace(func))


def _uses_annotations(
    specs: "tuple[_ArraySpec | _ScalarSpec | None, ...]",
) -> bool:
    return any(spec is not None for spec in specs)


def _split(
    args: tuple[typing.Any, ...],
) -> tuple[tuple[typing.Any, ...], tuple[typing.Any, ...]]:
    """Separate leading cast markers from the numba options."""
    if args and all(isinstance(a, _SPECS) for a in args):
        return args, ()

    return (), args


def _resolve(
    specs: tuple[typing.Any, ...],
) -> tuple[tuple[typing.Any, ...], list[_Caster | None]]:
    """Turn cast markers into numba arguments and per-parameter casters."""
    if not specs:
        return (), []

    if len(specs) == 1 and isinstance(specs[0], _Signature):
        signature = specs[0]
        signature_casters: list[_Caster | None] = [
            _elementwise_caster(arg.dtype) for arg in signature.args
        ]

        return (signature.numba_signature(),), signature_casters

    parameter_casters: list[_Caster | None] = [
        spec.caster() if spec is not None else None for spec in specs
    ]

    if all(spec is not None and spec.typed for spec in specs):
        types = tuple(spec.numba_type() for spec in specs if spec is not None)

        return (types,), parameter_casters

    return (), parameter_casters


def _autocast(
    compiled: typing.Any,
    casters: typing.Sequence[_Caster | None],
    func: typing.Callable[..., typing.Any],
) -> typing.Any:
    """Wrap ``compiled`` so its arguments are cast before being passed."""
    if not any(casters):
        return compiled

    names = list(inspect.signature(func).parameters)
    mapping = {
        name: caster for name, caster in zip(names, casters) if caster is not None
    }

    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        casted = tuple(
            (
                mapping[names[i]](value)
                if i < len(names) and names[i] in mapping
                else value
            )
            for i, value in enumerate(args)
        )
        casted_kwargs = {
            name: mapping[name](value) if name in mapping else value
            for name, value in kwargs.items()
        }

        return compiled(*casted, **casted_kwargs)

    wrapper.__numba__ = compiled  # type: ignore[attr-defined]

    return wrapper


def _check_vectorize_specs(specs: tuple[typing.Any, ...]) -> None:
    """Validate the eager signature of ``@cm.vectorize`` given positional specs."""
    if len(specs) == 1 and isinstance(specs[0], _Signature):
        return

    if any(getattr(spec, "typed", False) for spec in specs):
        raise TypeError(
            "Precompiled @cm.vectorize needs a full signature, e.g. "
            "cm.vectorize(cm.Scalar[float, [cm.Scalar[float]]])."
        )


def _vectorize_signature(
    func: typing.Callable[..., typing.Any],
    specs: "tuple[_ArraySpec | _ScalarSpec | None, ...]",
) -> tuple[typing.Any, ...]:
    """Build a complete signature for an annotated ``@cm.vectorize`` kernel."""
    if not any(spec is not None and spec.typed for spec in specs):
        return specs

    result = _return_spec(func)

    if (
        result is not None
        and result.typed
        and all(spec is not None and spec.typed for spec in specs)
    ):
        arguments = tuple(spec for spec in specs if spec is not None)

        return (_Signature(result, arguments),)

    raise TypeError(
        "Precompiled @cm.vectorize with annotations needs a typed return, e.g. "
        'def f(x: "Scalar[float]") -> "Scalar[float]": ...'
    )


class _Kernel:
    """Descriptor for a numba kernel whose options are resolved on first use.

    Compilation is deferred so a model can inject its compile options
    (``parallel``, ``fastmath``, ``cache``) at instantiation time. When the
    kernel is used standalone (module level, or accessed on the class) it
    compiles with the options given to the decorator.
    """

    __slots__ = (
        "_cache",
        "_func",
        "_make",
        "_options",
        "_parallel",
        "_parallel_mode",
    )

    def __init__(
        self,
        func: typing.Callable[..., typing.Any],
        make: typing.Callable[[dict[str, typing.Any]], typing.Any],
        options: dict[str, typing.Any],
        parallel: bool,
        parallel_mode: str | None,
    ) -> None:
        self._func = func
        self._make = make
        self._options = dict(options)
        self._parallel = parallel
        self._parallel_mode = parallel_mode
        self._cache: dict[tuple[typing.Any, ...], typing.Any] = {}

    @property
    def name(self) -> str:
        return getattr(self._func, "__name__", "kernel")

    def _resolved_options(
        self, config: dict[str, typing.Any] | None
    ) -> dict[str, typing.Any]:
        """Merge the decorator options with a model's compile config."""
        options = dict(self._options)
        config = config or {}

        if config.get("fastmath"):
            options["fastmath"] = True

        if config.get("cache"):
            options["cache"] = True

        if self._parallel_mode == "njit":
            # A ``precompile`` kernel must opt in *and* the model must allow it.
            enabled = self._parallel and (
                bool(config.get("parallel")) if config else True
            )
        elif self._parallel_mode == "target":
            # Elementwise/reduction ufuncs are parallel-safe by default.
            enabled = self._parallel or bool(config.get("parallel"))
        else:
            enabled = False

        if enabled:
            if self._parallel_mode == "njit":
                options["parallel"] = True
            else:
                options["target"] = "parallel"

        # ``njit`` is nopython by default; force it for the ufunc decorators.
        if self._parallel_mode == "target":
            options.setdefault("nopython", True)

        return options

    def compile(self, config: dict[str, typing.Any] | None = None) -> typing.Any:
        """Compile (or return the cached) kernel for ``config``."""
        options = self._resolved_options(config)
        key = tuple(sorted(options.items()))
        compiled = self._cache.get(key)

        if compiled is None:
            compiled = self._make(options)
            self._cache[key] = compiled

        return compiled

    def numba_object(self, config: dict[str, typing.Any] | None = None) -> typing.Any:
        """Return the underlying numba dispatcher/ufunc for ``config``."""
        compiled = self.compile(config)
        numba_obj = getattr(compiled, "__numba__", compiled)

        # Numba marks plain Python functions with a ``"py_func"`` string.
        return compiled if isinstance(numba_obj, str) else numba_obj

    def signatures(self, config: dict[str, typing.Any] | None = None) -> list[str]:
        """Return the compiled signatures (or types) for ``config``."""
        numba_obj = self.numba_object(config)
        signatures = getattr(numba_obj, "signatures", None)

        if signatures is None:
            signatures = getattr(numba_obj, "types", None)

        return [str(signature) for signature in signatures] if signatures else []

    def __get__(
        self, instance: typing.Any, owner: type[typing.Any] | None = None
    ) -> typing.Any:
        config = (
            getattr(instance, "_compile_options", None)
            if instance is not None
            else None
        )

        return self.compile(config)

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return self.compile()(*args, **kwargs)

    @property
    def __func__(self) -> "_Kernel":
        """Mimic ``staticmethod.__func__`` for introspection."""
        return self

    @property
    def __numba__(self) -> typing.Any:
        """The underlying numba object, compiled with the decorator options."""
        return self.numba_object()

    def __repr__(self) -> str:
        return f"cm.kernel({self.name})"


def _apply(
    func: typing.Callable[..., typing.Any],
    specs: tuple[typing.Any, ...],
    options: tuple[typing.Any, ...],
    options_map: dict[str, typing.Any],
    compiler: typing.Callable[..., typing.Any],
    *,
    vectorize_mode: bool = False,
    parallel_mode: str | None = None,
    parallel: bool = False,
) -> "_Kernel":
    """Build a deferred kernel, resolving the specs from arguments/annotations."""
    resolved: tuple[typing.Any, ...] = specs

    if not resolved and not options:
        annotated = _annotation_specs(func)

        if vectorize_mode and _uses_annotations(annotated):
            resolved = _vectorize_signature(func, annotated)
        else:
            resolved = annotated

    numba_args, casters = _resolve(resolved)

    def make(resolved_options: dict[str, typing.Any]) -> typing.Any:
        compiled = compiler(*numba_args, *options, **resolved_options)(func)

        return _autocast(compiled, casters, func)

    return _Kernel(func, make, options_map, parallel, parallel_mode)


def _template(
    compiler: typing.Callable[..., typing.Any],
    *,
    vectorize_mode: bool = False,
    parallel_mode: str | None = None,
    parallel: bool = False,
    **defaults: typing.Any,
) -> typing.Callable[..., typing.Any]:
    """Build a decorator for ``compiler`` supporting bare, option and spec calls."""

    def decorator(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        specs, options = _split(args)
        options_map = {**defaults, **kwargs}

        if len(options) == 1 and not kwargs and callable(options[0]):
            return _apply(
                options[0],
                specs,
                (),
                options_map,
                compiler,
                vectorize_mode=vectorize_mode,
                parallel_mode=parallel_mode,
                parallel=parallel,
            )

        def wrap(func: typing.Callable[..., typing.Any]) -> typing.Any:
            return _apply(
                func,
                specs,
                options,
                options_map,
                compiler,
                vectorize_mode=vectorize_mode,
                parallel_mode=parallel_mode,
                parallel=parallel,
            )

        return wrap

    return decorator


def vectorize(
    *args: typing.Any,
    fastmath: bool = False,
    parallel: bool = False,
    **kwargs: typing.Any,
) -> typing.Any:
    """Vectorize a static method, optionally precompiling it.

    Specs can be passed positionally (``cm.Array``, ``cm.Scalar[float, [...]]``)
    or declared as annotations in the decorated function::

        @cm.vectorize
        def affine(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]": ...

    Precompilation needs a full signature because numba requires the return
    type. ``fastmath`` and ``parallel`` can be forced on the kernel, and a model
    can also provide them for every kernel at instantiation. Any other option is
    forwarded as a keyword.
    """
    if args and all(isinstance(a, _SPECS) for a in args):
        _check_vectorize_specs(args)

    return _template(
        nb.vectorize,
        vectorize_mode=True,
        parallel_mode="target",
        parallel=parallel,
        fastmath=fastmath,
        **kwargs,
    )(*args)


def precompile(
    *args: typing.Any,
    fastmath: bool = False,
    parallel: bool = False,
    **kwargs: typing.Any,
) -> typing.Any:
    """Precompile a static method with `numba.njit`.

    Specs can be passed positionally (``cm.Array``, ``cm.Array[float, 2]``) or
    declared as annotations in the decorated function::

        @cm.precompile
        def hunt(X: "Array[float, 2]", a: "Scalar[float]"): ...

    ``fastmath`` and ``parallel`` map to the `numba.njit` options of the
    same name (``parallel`` also needs `numba.prange` in the body). A model
    can also enable them for every kernel at instantiation. Any other numba
    option (``cache=True``, ``boundscheck=False``, ...) is forwarded through
    keyword arguments.
    """
    return _template(
        nb.njit,
        parallel_mode="njit",
        parallel=parallel,
        fastmath=fastmath,
        **kwargs,
    )(*args)


def jit(
    *args: typing.Any,
    fastmath: bool = False,
    parallel: bool = False,
    **kwargs: typing.Any,
) -> typing.Any:
    """Deprecated alias of `precompile`."""
    warnings.warn(
        "cm.jit is deprecated; use cm.precompile instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return precompile(*args, fastmath=fastmath, parallel=parallel, **kwargs)


def _contract_element(value: typing.Any) -> "_ArraySpec | _ScalarSpec":
    """Normalize one ``Contract`` entry into an array or scalar spec."""
    if value is Ellipsis:
        return _ScalarSpec(float)

    if isinstance(value, _ArraySpec):
        if value.dtype is None:
            raise TypeError("Contract array specs need a dtype, e.g. Array[float].")

        return value

    if isinstance(value, _ScalarSpec):
        if value.dtype is None:
            raise TypeError("Contract scalar specs need a dtype, e.g. Scalar[float].")

        return value

    try:
        np.dtype(value)
    except TypeError:
        raise TypeError(f"Unsupported Contract spec: {value!r}.") from None

    return _ArraySpec(value, "n")


def _contract_atom(dims: tuple[str, ...]) -> str:
    return "()" if not dims else "(" + ",".join(dims) + ")"


def _contract_numba_type(
    spec: "_ArraySpec | _ScalarSpec", dims: tuple[str, ...]
) -> typing.Any:
    if isinstance(spec, _ScalarSpec):
        return nb.from_dtype(np.dtype(spec.dtype))

    return nb.types.Array(  # type: ignore[no-untyped-call]
        nb.from_dtype(np.dtype(spec.dtype)), len(dims), spec.layout
    )


class _ContractSpec:
    """Full eager signature for ``@cm.reduce`` (core types + layout)."""

    __slots__ = ("casters", "layout", "numba_types")

    def __init__(self, inputs: tuple[typing.Any, ...], output: typing.Any) -> None:
        elements = [_contract_element(item) for item in (*inputs, output)]

        if isinstance(elements[-1], _ScalarSpec):
            raise TypeError(
                "cm.Contract does not support a scalar output; use an Array[...]."
            )

        named = {
            dim
            for spec in elements
            if isinstance(spec, _ArraySpec)
            for dim in spec.dims or ()
        }
        counter = itertools.count()
        resolved: list[tuple[_ArraySpec | _ScalarSpec, tuple[str, ...]]] = []

        for spec in elements:
            if isinstance(spec, _ScalarSpec):
                resolved.append((spec, ()))
                continue

            if isinstance(spec.ndim, str):
                resolved.append((spec, spec.dims or ()))
                continue

            rank = 1 if spec.ndim is None else spec.ndim
            dims: list[str] = []

            for _ in range(rank):
                while True:
                    label = f"k{next(counter)}"

                    if label not in named:
                        break

                dims.append(label)
                named.add(label)

            resolved.append((spec, tuple(dims)))

        input_dims = {dim for _, dims in resolved[:-1] for dim in dims}
        missing = set(resolved[-1][1]) - input_dims

        if missing:
            raise TypeError(
                f"Contract output dimensions {sorted(missing)} do not appear "
                "in any input."
            )

        self.numba_types = tuple(
            _contract_numba_type(spec, dims) for spec, dims in resolved
        )
        atoms = [_contract_atom(dims) for _, dims in resolved]
        self.layout = ",".join(atoms[:-1]) + "->" + atoms[-1]
        self.casters = [spec.caster() for spec, _ in resolved[:-1]]

    def __repr__(self) -> str:
        return f"cm.Contract[{list(self.numba_types[:-1])!r}, {self.numba_types[-1]!r}]"


class Contract:
    """Signature marker for `reduce`, e.g. ``Contract[[...], out]``.

    Each input/output is a cast spec (``Array`` / ``Scalar``) or a plain dtype.
    A plain dtype is shorthand for ``Array[dtype, "n"]``; ``...`` is a scalar of
    default ``float`` dtype. ``Array[dtype]`` is a whole-array argument with a
    private dimension, and ``Array[dtype, "m,n"]`` names its dimensions::

        Contract[[float, float, ..., ..., ...], float]          # (n),(n),(),(),()->(n)
        Contract[[float, Array[float]], float]                   # (n),(k)->(n)
        Contract[[Array[float, "m,n"], Array[float, "n,k"]],
                 Array[float, "m,k"]]                            # (m,n),(n,k)->(m,k)
    """

    def __class_getitem__(cls, params: typing.Any) -> "_ContractSpec":
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError(
                "Contract expects Contract[[inputs...], output], e.g. "
                'Contract[[Array[float, "m,n"], Array[float, "n,k"]], '
                'Array[float, "m,k"]].'
            )

        inputs, output = params

        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, (list, tuple)):
            raise TypeError("The first Contract argument must be a list of specs.")

        return _ContractSpec(tuple(inputs), output)


def _parse_signatures(
    signatures: typing.Any,
) -> list[tuple[typing.Any, ...]]:
    """Normalize a flat list of specs into a list of core signatures."""
    if isinstance(signatures, _SPECS):
        signatures = [signatures]

    items = list(signatures)

    if not items:
        raise TypeError("cm.reduce requires at least one signature.")

    if all(isinstance(item, _SPECS) for item in items):
        return [tuple(items)]

    return [tuple(signature) for signature in items]


def _annotated_signature(
    func: typing.Callable[..., typing.Any],
) -> list[tuple[typing.Any, ...]]:
    """Derive a single core signature from the annotations of ``func``."""
    specs = _annotation_specs(func)

    if not _uses_annotations(specs):
        raise TypeError(
            "cm.reduce needs signatures or fully typed parameter annotations."
        )

    if any(spec is None or not spec.typed for spec in specs):
        raise TypeError(
            "cm.reduce requires fully typed parameter annotations, e.g. "
            'def clip(row: "Array[float, 1]", out: "Array[float, 1]"): ...'
        )

    return [tuple(spec for spec in specs if spec is not None)]


def _contract_from_annotations(
    func: typing.Callable[..., typing.Any],
) -> _ContractSpec:
    """Infer a `Contract` signature from the annotations of ``func``.

    Every parameter must be annotated; the last one is the output.
    """
    specs = _annotation_specs(func)

    if not _uses_annotations(specs):
        raise TypeError(
            "cm.reduce needs a Contract, explicit signatures, or annotations."
        )

    if any(spec is None for spec in specs):
        raise TypeError(
            "cm.reduce requires every parameter to be annotated, e.g. "
            "def matmul(a: \"Array[float, 'm,n']\", b: \"Array[float, 'n,k']\", "
            "out: \"Array[float, 'm,k']\"): ..."
        )

    resolved = [spec for spec in specs if spec is not None]

    if len(resolved) < 2:
        raise TypeError("cm.reduce needs at least one input and one output.")

    return _ContractSpec(tuple(resolved[:-1]), resolved[-1])


def _resolve_reduce(
    parsed: list[tuple[typing.Any, ...]],
    layout: str,
) -> tuple[list[tuple[typing.Any, ...]], list[_Caster]]:
    """Validate fully typed specs and build the numba types and casters."""
    for signature in parsed:
        for spec in signature:
            if not getattr(spec, "typed", False):
                raise TypeError(
                    "cm.reduce requires fully typed specs, e.g. "
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

    return ftylist, casters


def _count_inputs(layout: str) -> int:
    left = layout.split("->")[0].strip()

    if not left:
        return 0

    depth = 0
    count = 1

    for ch in left:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1

    return count


def reduce(
    signatures: typing.Any = None,
    layout: str | None = None,
    *,
    fastmath: bool = False,
    parallel: bool = False,
    **kwargs: typing.Any,
) -> typing.Any:
    """Build a generalized ufunc (``numba.guvectorize``) static method.

    Three additive forms are supported:

    * Legacy numba style: a list of core type specs plus a layout, e.g.
      ``[cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]]`` with
      ``"(n),()->(n)"``.
    * The `Contract` marker, which encodes the core types and layout::

          @cm.reduce(Contract[[Array[float, "m,n"],
                                     Array[float, "n,k"]],
                                    Array[float, "m,k"]])
          def matmul(a, b, out): ...

    * Fully typed annotations, with the layout inferred (the last parameter is
      the output)::

          @cm.reduce
          def matmul(a: "Array[float, 'm,n']",
                     b: "Array[float, 'n,k']",
                     out: "Array[float, 'm,k']"): ...

    ``fastmath`` maps to the numba option and ``parallel`` sets
    ``target="parallel"``; any other option is forwarded as a keyword. A model
    can also provide ``fastmath``/``parallel``/``cache`` at instantiation.
    """
    options = {**kwargs, "fastmath": fastmath}

    if layout is None and isinstance(signatures, str):
        signatures, layout = None, signatures

    func: typing.Callable[..., typing.Any] | None = None

    if callable(signatures) and not isinstance(signatures, _ContractSpec):
        func, signatures = signatures, None

    inferred = signatures is None and layout is None

    if not isinstance(signatures, _ContractSpec) and not inferred and layout is None:
        raise TypeError("cm.reduce requires a layout, e.g. '(n),()->(n)'.")

    def wrap(target: typing.Callable[..., typing.Any]) -> typing.Any:
        casters: list[_Caster]

        if isinstance(signatures, _ContractSpec):
            ftylist = [signatures.numba_types]
            compiled_layout = signatures.layout
            casters = signatures.casters
        elif inferred:
            contract = _contract_from_annotations(target)
            ftylist = [contract.numba_types]
            compiled_layout = contract.layout
            casters = contract.casters
        elif signatures is None:
            assert layout is not None
            ftylist, casters = _resolve_reduce(_annotated_signature(target), layout)
            compiled_layout = layout
        else:
            assert layout is not None
            ftylist, casters = _resolve_reduce(_parse_signatures(signatures), layout)
            compiled_layout = layout

        def make(resolved_options: dict[str, typing.Any]) -> typing.Any:
            compiled = nb.guvectorize(  # type: ignore[no-untyped-call]
                ftylist, compiled_layout, **resolved_options
            )(target)

            return _autocast(compiled, casters, target)

        return _Kernel(target, make, options, parallel, "target")

    if func is not None:
        return wrap(func)

    return wrap


def kernels(
    cls: type[typing.Any],
    compile_options: dict[str, typing.Any] | None = None,
) -> dict[str, list[str]]:
    """Return the numba kernels declared on ``cls``."""
    found: dict[str, list[str]] = {}

    for name, attr in vars(cls).items():
        if isinstance(attr, _Kernel):
            found[name] = attr.signatures(compile_options)
            continue

        func = attr.__func__ if isinstance(attr, staticmethod) else attr
        numba_obj = getattr(func, "__numba__", func)

        # Numba marks plain Python functions with a ``"py_func"`` string.
        if isinstance(numba_obj, str):
            numba_obj = func

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
    "Contract",
    "Scalar",
    "jit",
    "kernels",
    "precompile",
    "prepare",
    "reduce",
    "vectorize",
]
