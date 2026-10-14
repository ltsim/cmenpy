

class KernelSize:
    """Population size bounds: current size clamped to ``[min, max]``.

    Parameters
    ----------
    size : int
        Current population size.
    min_size : int, optional
        Smallest allowed size; defaults to ``size``.
    max_size : int, optional
        Largest allowed size; defaults to ``size``.

    Raises
    ------
    ValueError
        If ``size`` falls outside ``[min_size, max_size]``.

    Attributes
    ----------
    max : int
        Largest allowed size (the buffer is allocated to this many rows).
    min : int
        Smallest allowed size.
    size : int
        Current size.
    """
    def __init__(
        self,
        size: int,
        min_size: int | None = None,
        max_size: int | None = None,
    ):
        if min_size is None:
            min_size = size

        if max_size is None:
            max_size = size

        if size < min_size or size > max_size:
            raise ValueError("Size must be between 'min_size' and 'max_size'.")

        self.__size = size
        self.__min = min_size
        self.__max = max_size

    @property
    def max(self) -> int:
        return self.__max

    @property
    def min(self) -> int:
        return self.__min

    @property
    def size(self) -> int:
        return self.__size
