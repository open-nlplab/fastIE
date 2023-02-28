# -*- coding: UTF-8 -*- 
from dataclasses import dataclass, field
from typing import Sequence, Union, Callable, Generator, Dict

from fastNLP.io import DataBundle

from fastie.envs import get_flag
from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils.registry import Registry
from fastie.utils.hub import Hub

from functools import partial

NER = Registry('NER')
RE = Registry('RE')
EE = Registry('EE')


@dataclass
class BaseTaskConfig(BaseNodeConfig):
    cuda: bool = field(
        default=False,
        metadata=dict(
            help='Whether to use your NVIDIA graphics card to accelerate the '
            'process.',
            existence=True))
    load_model: str = field(
        default='',
        metadata=dict(help='Load the model from the path or model name. ',
                      existence=True))
    epochs: int = field(default=20,
                        metadata=dict(help='Total number of training epochs. ',
                                      existence='train'))


class BaseTask(BaseNode):
    """所有任务的基类.

    Args:
        :cuda (bool)[train,evaluate,inference]=False: 是否使用显卡加速训练.
    """
    _config = BaseTaskConfig()

    def __init__(self,
                 cuda: Union[bool, int, Sequence[int]] = False,
                 load_model: str = '',
                 epochs: int = 20,
                 **kwargs):
        BaseNode.__init__(self, **kwargs)
        self.cuda = cuda
        self.load_model = load_model
        self.epochs = epochs

        object.__setattr__(self, 'run', self.generate_run_func(self.run))

    def generate_run_func(self, run_func: Callable):

        def run(data_bundle: DataBundle):

            if self.load_model != '':
                if hasattr(self, 'load_state_dict'):
                    self.load_state_dict(Hub.load(self.load_model))

            def after_run_callback(run_func: Callable,
                                   data_bundle: DataBundle):
                # 主要作用是增加参数
                parameters_or_data: dict = run_func(data_bundle)
                base_parameters: Dict[str, Union[int, str, Sequence]] = dict()

                # cuda 相关参数
                if isinstance(self.cuda, bool):
                    if self.cuda:
                        base_parameters['device'] = 0
                    else:
                        base_parameters['device'] = 'cpu'
                elif isinstance(self.cuda, Sequence) and isinstance(
                        self.cuda[0], int):
                    base_parameters['device'] = self.cuda
                    parameters_or_data.update(base_parameters)
                if self.load_model != '':
                    if not hasattr(self, 'load_state_dict'):
                        parameters_or_data['model'].\
                            load_state_dict(Hub.load(self.load_model))
                if hasattr(self, 'state_dict'):
                    setattr(parameters_or_data['model'], 'fastie_state_dict',
                            self.state_dict)
                parameters_or_data['n_epochs'] = self.epochs
                return parameters_or_data

            if get_flag() is None:
                raise ValueError('You should set the flag first.')
            else:
                yield after_run_callback(run_func, data_bundle)

        return run

    def run(self, data_bundle: DataBundle):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if isinstance(self.run, Generator):
            return self.run
        else:
            return self.run(*args, **kwargs)
