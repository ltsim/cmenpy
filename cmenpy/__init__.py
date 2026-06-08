__version__ = "2026a0"

from cmenpy.bounds import Bounds
from cmenpy.epoch import EpochIteration, Epoch
from cmenpy.model import declare, template, template_check
from cmenpy.model.optimizer import FunctionOptimizerModel, TemplateOptimizerModel
from cmenpy.model.optimizer.variables import Argument, Variable
from cmenpy.population import PopulationSwarm
from cmenpy.tensor.swarm import TensorSwarm
from cmenpy.functools import is_best, is_worst, best_of, worst_of
from cmenpy.functools import sort_agents
from cmenpy.population.agent.attributes import Scalar, Vector, DefineAgent

__all__ = [
    "declare",
    "template",
    "template_check",
    "sort_agents",
    "is_best",
    "is_worst",
    "best_of",
    "worst_of",
    "Argument",
    "Variable",
    "Scalar",
    "Vector",
    "EpochIteration",
    "Epoch",
    "DefineAgent",
    "PopulationSwarm",
    "TensorSwarm",
    "FunctionOptimizerModel",
    "TemplateOptimizerModel",
]
