"""Deferred-compilation descriptor (`_Kernel`) for numba kernels."""

import typing


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
