# -*- coding: UTF-8 -*- 
from fastie.controller.BaseController import BaseController, CONTROLLER
from fastie.envs import set_flag
from fastie.node import BaseNodeConfig

from fastNLP import DataSet
from fastNLP.io import DataBundle
from fastNLP import Evaluator as FastNLP_Evaluator

from typing import Union, Sequence, Optional

from dataclasses import dataclass


@dataclass
class EvaluatorConfig(BaseNodeConfig):
    pass


@CONTROLLER.register_module('evaluator')
class Evaluator(BaseController):

    def __init__(self):
        super(Evaluator, self).__init__()
        set_flag('eval')

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str]]] = None):
        parameters_or_data = BaseController.run(self, parameters_or_data)
        if parameters_or_data is None:
            print('Evaluating tool do not allow task and dataset to be left '
                  'empty. ')
            exit(1)
        evaluator = FastNLP_Evaluator(**parameters_or_data)
        evaluator.run()
