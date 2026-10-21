# Version 2026b2

+ Numba is now an **optional backend**. The core install only requires NumPy;
  `pip install "cmenpy[numba]"` restores compilation. Declaring kernels no
  longer imports numba: the package is probed lazily and nothing is compiled
  unless a kernel actually asks for it.
+ The `numba` compile option controls **whether compilation takes place**:
  `MyOptimizer(numba=True)` demands compilation (`ImportError` with an install
  hint when numba is missing), `numba=False` runs the kernels uncompiled even
  when numba is installed, and omitting it compiles if and only if numba is
  available. It works on the model constructor and on each decorator
  (`@cm.precompile(numba=False)`), with the model option winning.
+ Uncompiled kernels keep their specs (arguments are still auto-cast) and run
  as pure NumPy/Python: `@cm.precompile` calls the plain function,
  `@cm.vectorize` uses `numpy.vectorize`, and `@cm.reduce` uses a
  layout-driven NumPy loop over the core signature.
+ `cm.prepare(model)` reports uncompiled kernels as `["<python>"]`.
+ Models accept a `prepare` constructor option: `MyOptimizer(numba=True,
  prepare=True)` compiles every class kernel during instantiation — the
  equivalent of calling `model.prepare()` right away — while `prepare=False`
  (the default) keeps lazy compilation and `model.prepare()` still available.
+ Kernels cast array-protocol arguments **automatically, even when declared
  without specs**: a buffer view such as `pop.X` reaches the kernel as the
  ndarray it wraps, and a call returns a NumPy array ready for the
  `pop.X << result` write-back. Plain arrays, NumPy scalars, lists and tuples
  pass through untouched; declared specs still control dtype and layout.
+ `from cmenpy import prange` binds numba's `prange` when the backend is
  installed and the builtin `range` otherwise, so `parallel` kernels stay
  importable in both modes.

---------------------------------------------------------------------

# Version 2026b1

+ `@cm.module` turns a class into a reusable **Operator Module** (a Ruby-style
  mixin). It marks the class, materializes its `cm.Constant` annotations and
  keeps it usable as a plain base: `@cm.template class GA(Selection)` inherits
  kernels, helper methods and fields from `Selection`. Name collisions resolve
  through the MRO, and `cm.prepare(model)` now reports kernels inherited from
  an Operator Module (compilation walks `cls.__mro__` instead of `vars(cls)`).

+ `cm.Constant[type, value]` declares a **fixed, read-only** field for
  `@cm.template` models and `@cm.module` Operator Modules. The value is checked
  at declaration, exposed through the class as a read-only descriptor
  (`self.name` / `Cls.name`), and is *never* constructor-bound — passing it to
  the model call raises `TypeError`. Constants flow into every implementing
  model via normal inheritance, so they work identically on a template and on
  an Operator Module.

+ `cm.cast(value, dtype=None, layout="A")` is the explicit way to turn a
  population view (`pop.X` / `pop.F`) or any array-like into a NumPy array. The
  views expose a `to_numpy()` conversion, and kernels, comparators and buffer
  writes unwrap them internally. Implicit casting via `np.asarray` is now
  **deprecated** (it still works, emitting a `DeprecationWarning`) and will be
  removed in a later release.

---------------------------------------------------------------------

# Version 2026b0

+ A decent and civilized version, compatible with GNU GPLv3
+ Numba decorators accept parameter annotations (`"Array[float, 2]"`,
  `"Scalar[float]"`, and mypy-clean `Annotated[..., cm.Array[...]]`), in
  addition to the positional `cm.Array` / `cm.Scalar` markers.
+ `cm.jit` is renamed to `cm.precompile` (both use `numba.njit`); `cm.jit`
  remains as a deprecated alias that emits a `DeprecationWarning`.
+ `cm.precompile` (and the `cm.jit` alias) expose the `numba.njit` options
  `fastmath` and `parallel`; other numba options are forwarded as keywords.
+ `cm.vectorize` and `cm.reduce` also accept `fastmath=True` and `parallel=True`
  (`parallel` maps to numba's `target="parallel"`).
+ Decorators return a deferred `_Kernel`; kernels compile on first use (or via
  `cm.prepare`). A `@cm.template` model configures `parallel`, `fastmath` and
  `cache` at instantiation and they propagate to its kernels.
+ `cm.parallel` is removed; use `cm.precompile(parallel=True)` standalone, or a
  model instantiated with `parallel=True` for kernels marked `parallel=True`.
+ `cm.guvectorize` is renamed to `cm.reduce` (generalized ufunc). The layout
  is inferred from typed annotations (`Array[float, 'm,n']`, last parameter is
  the output); `cm.Contract[[inputs...], output]` remains as the explicit
  signature marker.

---------------------------------------------------------------------

# Version 0.0.0

+ Experimental test version
+ Modeled by decorators
+ Implementation of basic and primitive structure
+ Population vectorization to leverage SIMD