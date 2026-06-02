__version__ = "2026"

from cmenpy.agent.attributes import Scalar, Vector, DefineAgent
from cmenpy.bounds import Bounds
from cmenpy.epoch import EpochIteration, Epoch
from cmenpy.model import declare, template, template_check
from cmenpy.model.optimizer import FunctionOptimizerModel, TemplateOptimizerModel
from cmenpy.model.optimizer.variables import Argument, Variable
from cmenpy.population import PopulationManager, sort_agents
from cmenpy.population.functools import is_best, is_worst, best_of, worst_of

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
    "PopulationManager",
    "FunctionOptimizerModel",
    "TemplateOptimizerModel",
]
