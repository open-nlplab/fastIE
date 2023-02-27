from fastie.controller.BaseController import BaseController, CONTROLLER
from fastie.controller.inference import Inference
from fastie.node import BaseNodeConfig
from fastie.envs import set_flag

from dataclasses import dataclass, field

from typing import Union, Sequence, Optional

from fastNLP import DataSet
from fastNLP.io import DataBundle


@dataclass
class InteractorConfig(BaseNodeConfig):
    log: str = field(
        default='',
        metadata={
            'help':
            'What file to write the interactive log to. If this is not set, '
            'the log will not be written. ',
            'existence':
            'interact'
        })


@CONTROLLER.register_module('interactor')
class Interactor(BaseController):

    def __init__(self, log: Optional[str] = None):
        super(Interactor, self).__init__()
        self.log = log
        if self.log is not None:
            self.inference = Inference(save_path=self.log, verbose=True)
        else:
            self.inference = Inference(verbose=True)
        set_flag('interact')

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str]]] = None):
        parameters_or_data = BaseController.run(self, parameters_or_data)
        if parameters_or_data is None:
            print('Interacting tool do not allow task and dataset to be left '
                  'empty. ')
            exit(1)
        return self.inference(parameters_or_data=parameters_or_data)
