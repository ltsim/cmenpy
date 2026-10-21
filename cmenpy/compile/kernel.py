"""Deferred-compilation descriptor (`_Kernel`) for numba kernels."""

import typing

from cmenpy.compile import backend

#: Signature reported by `kernels`/`prepare` for a kernel running uncompiled.
PYTHON_SIGNATURE = "<python>"


def _numba_object(obj: typing.Any) -> typing.Any:
    """Return the numba dispatcher/ufunc behind ``obj``, unwrapping autocast.

    Numba marks plain Python functions with a ``"py_func"`` string, so the
    original object is returned when there is nothing compiled underneath.
    """
    numba_obj = getattr(obj, "__numba__", obj)

    return obj if isinstance(numba_obj, str) else numba_obj


def _signatures_of(obj: typing.Any) -> list[str] | None:
    """Return the signatures a compiled kernel reports, or ``None`` if not compiled.

    Reads ``signatures`` (a dispatcher) or ``types`` (a ufunc) off the numba
    object behind ``obj``; an empty list means "compiled, nothing yet".
    """
    numba_obj = _numba_object(obj)
    signatures = getattr(numba_obj, "signatures", None)

    if signatures is None:
        signatures = getattr(numba_obj, "types", None)

    return None if signatures is None else [str(signature) for signature in signatures]


class _Kernel:
    """Descriptor for a numba kernel whose options are resolved on first use.

    Compilation is deferred so a model can inject its compile options
    (``numba``, ``parallel``, ``fastmath``, ``cache``) at instantiation time.
    When the kernel is used standalone (module level, or accessed on the class)
    it compiles with the options given to the decorator.

    The ``numba`` option controls *whether compilation takes place at all*:
    ``numba=False`` runs the pure-Python flavor (auto-casting still applies),
    ``numba=True`` demands compilation and raises ``ImportError`` when numba
    is not installed. When nobody expresses an opinion, compilation happens
    if and only if numba is available.
    """

    __slots__ = (
        "_cache",
        "_func",
        "_make",
        "_make_python",
        "_options",
        "_parallel",
        "_parallel_mode",
        "_python",
    )

    def __init__(
        self,
        func: typing.Callable[..., typing.Any],
        make: typing.Callable[[dict[str, typing.Any]], typing.Any],
        make_python: typing.Callable[[], typing.Any],
        options: dict[str, typing.Any],
        parallel: bool,
        parallel_mode: str | None,
    ) -> None:
        self._func = func
        self._make = make
        self._make_python = make_python
        self._options = dict(options)
        self._parallel = parallel
        self._parallel_mode = parallel_mode
        self._cache: dict[tuple[typing.Any, ...], typing.Any] = {}
        self._python: typing.Any = None

    @property
    def name(self) -> str:
        return getattr(self._func, "__name__", "kernel")

    def _wants_python(self, config: dict[str, typing.Any] | None) -> bool:
        """Decide whether the kernel runs uncompiled for ``config``.

        A model's explicit ``numba`` option wins over the decorator's; if
        neither expresses an opinion, the kernel compiles when numba is
        available and runs as plain Python when it is not.
        """
        if config is not None and "numba" in config:
            return not bool(config["numba"])

        if "numba" in self._options:
            return not bool(self._options["numba"])

        return not backend.HAS_NUMBA

    def _require_numba(self, config: dict[str, typing.Any] | None) -> None:
        """Raise when compilation was explicitly requested but is impossible."""
        explicit = config is not None and config.get("numba")

        if explicit is None:
            explicit = self._options.get("numba")

        if explicit and not backend.HAS_NUMBA:
            raise ImportError(backend._INSTALL_HINT)

    def _resolved_options(
        self, config: dict[str, typing.Any] | None
    ) -> dict[str, typing.Any]:
        """Merge the decorator options with a model's compile config."""
        options = {key: value for key, value in self._options.items() if key != "numba"}
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

    def python(self) -> typing.Any:
        """Return the cached pure-Python flavor of the kernel."""
        if self._python is None:
            self._python = self._make_python()

        return self._python

    def compile(self, config: dict[str, typing.Any] | None = None) -> typing.Any:
        """Compile (or return the cached) kernel for ``config``.

        With the ``numba`` option off, return the pure-Python flavor instead
        of compiling.
        """
        if self._wants_python(config):
            return self.python()

        self._require_numba(config)
        options = self._resolved_options(config)
        key = tuple(sorted(options.items()))
        compiled = self._cache.get(key)

        if compiled is None:
            compiled = self._make(options)
            self._cache[key] = compiled

        return compiled

    def numba_object(self, config: dict[str, typing.Any] | None = None) -> typing.Any:
        """Return the underlying numba dispatcher/ufunc for ``config``."""
        return _numba_object(self.compile(config))

    def signatures(self, config: dict[str, typing.Any] | None = None) -> list[str]:
        """Return the compiled signatures (or types) for ``config``.

        An uncompiled kernel reports the ``"<python>"`` marker.
        """
        if self._wants_python(config):
            return [PYTHON_SIGNATURE]

        return _signatures_of(self.compile(config)) or []

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
