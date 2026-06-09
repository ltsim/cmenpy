from cmenpy.kernel import KernelBuffer


class BaseIterator:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def best(self):
        return self.__buffer.idx.best

    @property
    def worst(self):
        return self.__buffer.idx.worst

    @property
    def sort(self):
        return self.__buffer.idx.sort
