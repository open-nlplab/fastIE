from fastie.controller import BaseController, CONTROLLER
from fastie.envs import set_flag
from fastie.node import BaseNodeConfig

from fastNLP import DataSet
from fastNLP.io import DataBundle
from fastNLP import Evaluator as FastNLP_Evaluator

from typing import Union, Sequence

from dataclasses import dataclass


@dataclass
class EvaluatorConfig(BaseNodeConfig):
    pass


@CONTROLLER.register_module('evaluator')
class Evaluator(BaseController):

    def __init__(self):
        BaseController.__init__(self)
        set_flag('eval')

    def run(self,
            parameters_or_data: Union[dict, DataBundle, DataSet, str,
                                      Sequence[str]] = dict()):
        parameters_or_data = BaseController.run(self, parameters_or_data)
        evaluator = FastNLP_Evaluator(**parameters_or_data)
        evaluator.run()
