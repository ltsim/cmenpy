from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent import AgentGenerator
from cmenpy.population.attributes import TensorOperations
from cmenpy.population.iterator.base import BaseIterator
from cmenpy.population.dynamic.base import DynamicBase
from cmenpy.population.operations import StreamOperations


class PopulationSwarm(
    StreamOperations, TensorOperations, BaseIterator, DynamicBase, AgentGenerator
):
    def __init__(self, buffer: KernelBuffer) -> None:
        StreamOperations.__init__(self, buffer)
        TensorOperations.__init__(self, buffer)
        BaseIterator.__init__(self, buffer)
        DynamicBase.__init__(self, buffer)
        AgentGenerator.__init__(self, buffer)

        self.__buffer = buffer
