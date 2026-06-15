import typing

import numpy as np
import numpy.typing as npt

ArrayType = npt.NDArray[np.number[typing.Any]]
ArrayFloatType = npt.NDArray[np.floating[typing.Any]]
ArrayIntegerType = npt.NDArray[np.integer[typing.Any]]
ArrayBoolType = npt.NDArray[np.bool_]
ArrayObjectType = npt.NDArray[np.object_]
