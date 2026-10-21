"""Pure-Python execution of ``cm.reduce`` generalized ufuncs (no numba needed).

When a kernel runs without compilation, ``cm.vectorize`` falls back to
``numpy.vectorize`` (handled in ``cmenpy.compile.low.decorate``) and ``cm.reduce``
falls back to the layout-driven emulator here: the core-signature layout is
parsed, the function is applied to every core slice while NumPy broadcasts the
loop dimensions, and the output is allocated exactly as numba's ``guvectorize``
would allocate it.
"""

import functools
import typing

import numpy as np


def _split_atoms(text: str) -> list[str]:
    """Return the contents of the depth-1 parenthesized groups of ``text``."""
    groups: list[str] = []
    depth = 0
    buffer = ""

    for char in text:
        if char == "(":
            depth += 1

            if depth == 1:
                buffer = ""
            else:
                buffer += char
        elif char == ")":
            depth -= 1

            if depth == 0:
                groups.append(buffer)
            else:
                buffer += char
        elif depth >= 1:
            buffer += char

    return groups


def _atom_dims(atom: str) -> tuple[str, ...]:
    """Return the dimension labels of one core signature atom."""
    return tuple(
        label for label in (part.strip("() ") for part in atom.split(",")) if label
    )


def layout_atoms(layout: str) -> list[tuple[str, ...]]:
    """Split a core signature such as ``"(m,n),(n,k)->(m,k)"`` into atoms.

    The last atom is the output; scalars are empty tuples.
    """
    left, separator, right = layout.partition("->")

    if not separator:
        raise ValueError(f"Invalid core signature layout: {layout!r}.")

    return [_atom_dims(atom) for atom in _split_atoms(left) + _split_atoms(right)]


def gufunc(
    func: typing.Callable[..., typing.Any],
    layout: str,
) -> typing.Callable[..., typing.Any]:
    """Build a pure-Python generalized ufunc for ``func`` from ``layout``.

    Calling the result applies ``func`` to one core slice per loop index; the
    output array is allocated from the broadcast loop shape and the core
    dimensions named in the layout, and is also returned.
    """
    atoms = layout_atoms(layout)
    input_atoms = atoms[:-1]
    output_dims = atoms[-1]

    @functools.wraps(func)
    def call(
        *args: typing.Any, out: np.typing.NDArray[typing.Any] | None = None
    ) -> typing.Any:
        if len(args) != len(input_atoms):
            raise TypeError(
                f"{getattr(func, '__name__', 'kernel')} takes "
                f"{len(input_atoms)} array arguments, got {len(args)}."
            )

        dim_sizes: dict[str, int] = {}
        loop_shape: tuple[int, ...] | None = None
        prepared: list[tuple[typing.Any, tuple[int, ...], tuple[int, ...]]] = []

        for arg, dims in zip(args, input_atoms):
            rank = len(dims)

            if rank == 0:
                prepared.append((arg, (), ()))
                continue

            array = np.asarray(arg)

            if array.ndim < rank:
                raise ValueError(
                    f"Input has {array.ndim} dimensions but the layout needs "
                    f"{rank} core dimensions."
                )

            core = array.shape[-rank:]
            loop = array.shape[:-rank]

            for dim, size in zip(dims, core):
                if dim_sizes.setdefault(dim, size) != size:
                    raise ValueError(
                        f"Core dimension '{dim}' is {dim_sizes[dim]} in one "
                        f"input and {size} in another."
                    )

            loop_shape = (
                loop if loop_shape is None else np.broadcast_shapes(loop_shape, loop)
            )
            prepared.append((array, core, loop))

        if loop_shape is None:
            loop_shape = ()

        missing = [dim for dim in output_dims if dim not in dim_sizes]

        if missing:
            raise ValueError(f"Output dimensions {missing} do not appear in any input.")

        out_core = tuple(dim_sizes[dim] for dim in output_dims)

        if out is None:
            out = np.empty(
                loop_shape + out_core,
                dtype=np.result_type(*[arg for arg, _, _ in prepared], float),
            )
        else:
            out = np.asarray(out)

        for index in np.ndindex(loop_shape):
            output_index: typing.Any = (
                index + (slice(None),) * len(out_core) if loop_shape else ...
            )
            core_args = []

            for (arg, core, loop), dims in zip(prepared, input_atoms):
                if len(dims) == 0:
                    core_args.append(arg)
                    continue

                array = arg

                if loop != loop_shape:
                    array = np.broadcast_to(array, loop_shape + core)

                core_args.append(array[index] if loop_shape else array)

            func(*core_args, out[output_index])

        return out

    return call
