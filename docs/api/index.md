# API Reference

Complete autodoc for the public surface of `cmenpy`, organized by kind. Every
class, function and type alias the framework exposes is documented here, and
each symbol links to the page that documents it.

| Browse | Contents |
| ------ | -------- |
| **[Classes](classes.md)** | Optimizers, field markers, population, agents, buffer, objectives, search space, runtime, protocols, aliases |
| **[Functions](functions.md)** | Declaration decorators, kernel decorators, signature markers, builders, introspection, comparators |
| **[Modules](modules.md)** | One entry per subpackage, with its docstring and links to its symbols |

!!! note "How to read this reference"
    Documentation is generated from the source via
    [mkdocstrings](https://mkdocstrings.github.io/). Signatures are shown with
    their annotations, and class initializers are merged into the class entry.

## Symbol index

| Symbol | Kind | Page |
| ------ | ---- | ---- |
| `Array`, `Scalar` | compile markers | [Functions](functions.md#signature-markers) |
| `Contract` | compile marker | [Functions](functions.md#signature-markers) |
| `precompile`, `vectorize`, `reduce` | decorators | [Functions](functions.md#kernel-decorators) |
| `jit` | decorator (deprecated) | [Functions](functions.md#kernel-decorators) |
| `kernels`, `prepare` | introspection | [Functions](functions.md#introspection) |
| `declare`, `template` | model decorators | [Functions](functions.md#declaration-decorators) |
| `module` | Operator Module decorator | [Functions](functions.md#declaration-decorators) |
| `vectorized_target` | decorator | [Functions](functions.md#declaration-decorators) |
| `declare_from_func`, `declare_from_class`, `build_context_manager` | builders | [Functions](functions.md#builders-and-factories) |
| `create_bounds`, `init_buffer` | factories | [Functions](functions.md#builders-and-factories) |
| `is_best`, `is_worst` | comparators | [Functions](functions.md#comparators) |
| `best_index`, `worst_index` | ranking helpers | [Functions](functions.md#comparators) |
| `Argument`, `Constant`, `Variable` | field markers | [Classes](classes.md#field-markers) |
| `BaseOptimizer` | abstract base | [Classes](classes.md#optimizers) |
| `FunctionOptimizerModel`, `TemplateOptimizerModel` | models | [Classes](classes.md#optimizers) |
| `ModelProtocol`, `AlgorithmFunction`, `CallableFunction` | protocols | [Classes](classes.md#protocols) |
| `FunctionParamsArguments` | arguments holder | [Classes](classes.md#support-types-and-aliases) |
| `ExceptionTypeParams`, `ExceptionValueParams` | exceptions | [Classes](classes.md#support-types-and-aliases) |
| `PopulationSwarm` | population handle | [Classes](classes.md#population) |
| `StreamOperations` | operator set | [Classes](classes.md#population) |
| `ComputeOperator`, `AssignOperator`, `AccessOperator`, `ExtractOperation`, `SwapOperation` | operators | [Classes](classes.md#population) |
| `ComputePipeline`, `AssignPipeline` | per-row targets | [Classes](classes.md#population) |
| `StreamOperator`, `ReaderOperator`, `WriterOperator`, `VectorizableOperator`, `PipelineOperator` | operator ABCs | [Classes](classes.md#population) |
| `BufferMatrixOperations`, `BufferMatrixFitness`, `BufferMatrixSolutions` | matrix views | [Classes](classes.md#population) |
| `BufferMatrixAbstract`, `BufferMatrixAccess`, `BufferMatrixCompute` | matrix ABCs | [Classes](classes.md#population) |
| `DynamicBase`, `BaseIterator` | population bases | [Classes](classes.md#population) |
| `BaseAgent`, `ImmutableAgent`, `MutableAgent`, `AgentTemplate` | agents | [Classes](classes.md#agents) |
| `KernelBuffer` | buffer | [Classes](classes.md#buffer) |
| `KernelMask`, `KernelSize` | buffer support | [Classes](classes.md#buffer) |
| `BufferIndex` | ranking | [Classes](classes.md#buffer) |
| `Target`, `TargetFunction` | objective | [Classes](classes.md#objectives) |
| `VectorizedTarget` | vectorized objective | [Classes](classes.md#objectives) |
| `MaxNumberOfCalls` | exception | [Classes](classes.md#objectives) |
| `Bounds` | search space | [Classes](classes.md#search-space) |
| `Context`, `MainContextManager` | run context | [Classes](classes.md#runtime) |
| `EpochIteration` | epoch iterator | [Classes](classes.md#runtime) |
| `Tracker`, `EpochHistory` | history | [Classes](classes.md#runtime) |
| `DefaultGenerator` | RNG | [Classes](classes.md#runtime) |
| `NDArrayType`, `ArrayFloatType`, `ArrayIntegerType` | array aliases | [Classes](classes.md#support-types-and-aliases) |
| `ScalarType`, `SenseType` | scalar/option aliases | [Classes](classes.md#support-types-and-aliases) |
