from fastie.controller.BaseController import BaseController, CONTROLLER
from fastie.envs import set_flag
from fastie.node import BaseNodeConfig

from fastNLP import DataSet
from fastNLP.io import DataBundle
from fastNLP import Trainer as FastNLP_Trainer

from typing import Union, Sequence, Optional

from dataclasses import dataclass


@dataclass
class TrianerConfig(BaseNodeConfig):
    pass


@CONTROLLER.register_module('trainer')
class Trainer(BaseController):

    def __init__(self):
        super(Trainer, self).__init__()
        set_flag('train')

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str]]] = None):
        parameters_or_data = BaseController.run(self, parameters_or_data)
        if parameters_or_data is None:
            print(
                'Training tool do not allow task and dataset to be left '
                'empty. '
            )
            exit(1)
        trainer = FastNLP_Trainer(**parameters_or_data)
        trainer.run()
