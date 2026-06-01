__version__ = "2026"

from cmenpy.bounds import Bounds
from cmenpy.population import Population
from cmenpy.model import declare, template, template_check
from cmenpy.model.optimizer.variables import Argument, Variable
from cmenpy.model.optimizer import FunctionOptimizerModel, TemplateOptimizerModel
from cmenpy.agent.attributes import Scalar, Vector

__all__ = [
    "declare",
    "template",
    "template_check",
    "Argument",
    "Variable",
    "Scalar",
    "Vector",
    "Population",
    "FunctionOptimizerModel",
    "TemplateOptimizerModel",
]
