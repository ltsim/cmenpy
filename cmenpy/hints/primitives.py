import typing

import numpy as np

#: Type variable bound to a NumPy scalar, used for scalar bounds/values.
ScalarType = typing.TypeVar("ScalarType", bound=np.number[typing.Any])
