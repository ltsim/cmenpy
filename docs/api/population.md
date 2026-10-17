# Population

The swarm handle and everything layered on top of the kernel buffer: stream
operators, matrix views, roster/iteration bases and the agent classes.

::: cmenpy.PopulationSwarm

## Stream operations

::: cmenpy.population.operations.StreamOperations

::: cmenpy.population.operations.ComputeOperator

::: cmenpy.population.operations.AssignOperator

::: cmenpy.population.operations.AccessOperator

::: cmenpy.population.operations.ExtractOperation

::: cmenpy.population.operations.SwapOperation

::: cmenpy.population.operations.compute.ComputePipeline

::: cmenpy.population.operations.assign.AssignPipeline

## Operator base classes

::: cmenpy.population.operations.stream.base.StreamOperator

::: cmenpy.population.operations.stream.reader.ReaderOperator

::: cmenpy.population.operations.stream.writer.WriterOperator

::: cmenpy.population.operations.stream.vectorizable.VectorizableOperator

::: cmenpy.population.operations.stream.vectorizable.PipelineOperator

## Matrix views

::: cmenpy.population.attributes.base.BufferMatrixOperations

::: cmenpy.population.attributes.base.BufferMatrixFitness

::: cmenpy.population.attributes.base.BufferMatrixSolutions

::: cmenpy.population.attributes.matrix.BufferMatrixAbstract

::: cmenpy.population.attributes.matrix.mixin.BufferMatrixAccess

::: cmenpy.population.attributes.matrix.mixin.BufferMatrixCompute

## Population bases

::: cmenpy.population.dynamic.base.DynamicBase

::: cmenpy.population.iterator.base.BaseIterator

## Agents

::: cmenpy.population.agent.BaseAgent

::: cmenpy.population.agent.ImmutableAgent

::: cmenpy.population.agent.MutableAgent

::: cmenpy.hints.protocol.agent.AgentTemplate
