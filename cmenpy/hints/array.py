import typing

import numpy as np
import numpy.typing as npt

#: Alias for a NumPy array of any numeric dtype.
NDArrayType = npt.NDArray[np.number[typing.Any]]
#: Alias for a NumPy array of floating dtype.
ArrayFloatType = npt.NDArray[np.floating[typing.Any]]
#: Alias for a NumPy array of integer dtype.
ArrayIntegerType = npt.NDArray[np.integer[typing.Any]]
