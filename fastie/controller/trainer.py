from fastie.controller import BaseController, CONTROLLER
from fastie.envs import set_flag
from fastie.node import BaseNodeConfig

from fastNLP import DataSet
from fastNLP.io import DataBundle
from fastNLP import Trainer as FastNLP_Trainer

from typing import Union, Sequence

from dataclasses import dataclass

@dataclass
class TrianerConfig(BaseNodeConfig):
    pass

@CONTROLLER.register_module("trainer")
class Trainer(BaseController):
    def __init__(self):
        BaseController.__init__(self)
        set_flag("train")

    def run(self,
            parameters_or_data: Union[dict, DataBundle, DataSet, str, Sequence[str]] = {}):
        parameters_or_data = BaseController.run(self, parameters_or_data)
        trainer = FastNLP_Trainer(**parameters_or_data)
        trainer.run()
