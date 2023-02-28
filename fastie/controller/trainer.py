# -*- coding: UTF-8 -*- 
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
    save_model: str = field(default="",
                            metadata={
                                "help": "Path to save the model after training. "
                                        "If this parameter is not set, "
                                        "the path is not saved. ",
                                "existence": "train"
                            })


@CONTROLLER.register_module('trainer')
class Trainer(BaseController):
    _config = TrainerConfig()
    _help = "Trainer for FastIE "

    def __init__(self,
                 save_model: str = ""):
        super(Trainer, self).__init__()
        self.save_model = save_model
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
        if self.save_model != "":
            if hasattr(trainer.model, "fastie_state_dict"):
                model = trainer.model.fastie_state_dict()
                Hub.save(self.save_model, model)
            else:
                model = trainer.model.state_dict()
                Hub.save(self.save_model, model)
        return model
