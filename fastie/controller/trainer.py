from fastie.controller.BaseController import BaseController, CONTROLLER
from fastie.envs import set_flag
from fastie.node import BaseNodeConfig
from fastie.utils.hub import Hub

from fastNLP import DataSet
from fastNLP.io import DataBundle
from fastNLP import Trainer as FastNLP_Trainer

from typing import Union, Sequence, Optional

from dataclasses import dataclass, field


@dataclass
class TrainerConfig(BaseNodeConfig):
    pass


@CONTROLLER.register_module('trainer')
class Trainer(BaseController):
    _config = TrainerConfig()
    _help = 'Trainer for FastIE '

    def __init__(self):
        super(Trainer, self).__init__()
        set_flag('train')

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str]]] = None):
        parameters_or_data = BaseController.run(self, parameters_or_data)
        if parameters_or_data is None:
            print('Training tool do not allow task and dataset to be left '
                  'empty. ')
            exit(1)
        trainer = FastNLP_Trainer(**parameters_or_data)
        trainer.run()
        model: dict = {}
        if hasattr(trainer.model, 'fastie_save_step'):
            trainer.model.fastie_save_step()
        return model
