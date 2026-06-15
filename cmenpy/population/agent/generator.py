from cmenpy.kernel import KernelBuffer


class AgentIterator:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer


class AgentGenerator:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def agent(self) -> AgentIterator:
        return AgentIterator(self.__buffer)
