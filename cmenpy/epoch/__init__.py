import time


class Epoch:
    def __init__(self, e: int):
        self.__e: int = e

    def __int__(self):
        return self.__e

    def __float__(self):
        return float(self.__e)

    def __repr__(self):
        return f"Epoch({self.__e})"


class EpochIteration:
    def __init__(
        self,
        epochs: int,
    ):
        if epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")

        self.__epochs: int = epochs
        self.__current_epoch: int = 0
        self.__cpu_time: float = 0

    def __iter__(self):
        for i in range(0, self.__epochs):
            self.__current_epoch = i

            start = time.process_time()
            yield Epoch(i)
            self.__cpu_time = time.process_time() - start

    @property
    def current(self):
        return self.__current_epoch

    @property
    def max(self) -> int:
        return self.__epochs

    @property
    def cpu_time(self):
        return self.__cpu_time
