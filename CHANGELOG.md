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