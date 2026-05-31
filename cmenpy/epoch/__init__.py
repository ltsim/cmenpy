class EpochIteration:
    def __init__(
        self,
        epochs: int,
    ):
        if epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")

        self.__epochs: int = epochs
        self.__current_epoch: int = 0

    def __iter__(self):
        for i in range(0, self.__epochs):
            self.__current_epoch = i
            yield i

    @property
    def current(self):
        return self.__current_epoch

    @property
    def max_epochs(self):
        return self.__epochs
