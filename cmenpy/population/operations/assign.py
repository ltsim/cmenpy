from cmenpy.population.operations.stream import WriterOperator, VectorizableOperator
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType, ArrayIntegerType


class AssignOperator(VectorizableOperator):

    def __init__(
        self,
        mask: ArrayIntegerType,
        buffer: NDArrayType,
        target: TargetFunction,
    ):
        super().__init__(buffer.shape)
        self.__mask = mask
        self.__buffer = buffer
        self.__target = target

    def __setitem__(self, item, value) -> None:
        if self.rows != len(value):
            raise IndexError(
                "The buffer cannot be assigned a value; the sizes do not match."
            )

        self.__buffer[item] = value

    def __imatmul__(self, other: NDArrayType) -> "VectorizableOperator":
        if self.shape != other.shape:
            raise IndexError(
                "The buffer cannot be assigned a value; the sizes do not match."
            )

        self.__buffer[:] = other

        return self

    def __matmul__(self, other: NDArrayType) -> None:
        if self.shape != other.shape:
            raise IndexError(
                "The buffer cannot be assigned a value; the sizes do not match."
            )

        self.__buffer[:] = other

    def __repr__(self) -> str:
        return "Assign<VectorizableOperator>()"
