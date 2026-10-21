"""The ``Contract`` signature marker for `cm.reduce`."""

import typing

from cmenpy.compile.reduce.contract_spec import _ContractSpec


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
        """Build a ``_ContractSpec`` from ``Contract[[inputs...], output]``."""
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
