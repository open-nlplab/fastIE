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
    save_model: str = field(
        default='',
        metadata=dict(help='The path to save the model in last epoch '
                           '(Only available for train). ',
                      existence="train"))
    epochs: int = field(default=20,
                        metadata=dict(help='Total number of training epochs. ',
                                      existence='train'))
    topk: int = field(default=0,
                      metadata=dict(
                          help='Save the top-k models according to metric. '
                          '(Only available for train). ',
                          existence='train'
                      ))


class BaseTask(BaseNode):
    """所有任务的基类.

    Args:
        :cuda (bool)[train,evaluate,inference]=False: 是否使用显卡加速训练.
    """
    _config = BaseTaskConfig()

    def __init__(self,
                 cuda: Union[bool, int, Sequence[int]] = False,
                 load_model: str = '',
                 save_model: str = '',
                 epochs: int = 20,
                 topk: int = 0,
                 **kwargs):
        BaseNode.__init__(self, **kwargs)
        self.cuda = cuda
        self.load_model = load_model
        self.save_model = save_model
        self.epochs = epochs
        self.topk = topk
        object.__setattr__(self, 'run', self.generate_run_func(self.run))

    def generate_run_func(self, run_func: Callable):

        def run(data_bundle: DataBundle):

            def run_warp(run_func: Callable,
                                   data_bundle: DataBundle):
                # 如果存在自定义的模型加载策略
                if self.load_model != '':
                    if hasattr(self, 'load_state_dict'):
                        self.load_state_dict(Hub.load(self.load_model))
                # 任务参数
                parameters_or_data: dict = run_func(data_bundle)
                base_parameters: Dict[str, Union[int, str, Sequence]] = dict()
                # 如果不存在自定义的模型加载策略
                if self.load_model != '':
                    if not hasattr(self, 'load_state_dict'):
                        parameters_or_data['model'].\
                            load_state_dict(Hub.load(self.load_model))
                # cuda 相关参数
                if isinstance(self.cuda, bool):
                    if self.cuda:
                        base_parameters['device'] = 0
                    else:
                        base_parameters['device'] = 'cpu'
                elif isinstance(self.cuda, Sequence) and isinstance(
                        self.cuda[0], int) or isinstance(self.cuda, int):
                    base_parameters['device'] = self.cuda
                # 保存模型相关
                if self.save_model != '':
                    # 如果存在自定义模型保存策略
                    if hasattr(self, 'state_dict'):
                        def fastie_save_step():
                            Hub.save(self.save_model, self.state_dict())
                        setattr(parameters_or_data['model'], 'fastie_save_step',
                            fastie_save_step)
                    # 如果不存在自定义模型保存策略
                    else:
                        def fastie_save_step():
                            Hub.save(self.save_model,
                                     parameters_or_data['model'].state_dict())
                        setattr(parameters_or_data['model'], 'fastie_save_step',
                            fastie_save_step)
                # 不保存模型
                else:
                    def fastie_save_step():
                        pass
                    setattr(parameters_or_data['model'], 'fastie_save_step',
                        fastie_save_step)
                # 训练轮数
                base_parameters['n_epochs'] = self.epochs
                parameters_or_data.update(base_parameters)
                return parameters_or_data

            if get_flag() is None:
                raise ValueError('You should set the flag first.')
            else:
                while True:
                    yield run_warp(run_func, data_bundle)

        return run

    def run(self, data_bundle: DataBundle):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if isinstance(self.run, Generator):
            return self.run
        else:
            return self.run(*args, **kwargs)
