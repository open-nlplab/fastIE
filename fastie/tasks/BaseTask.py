from dataclasses import dataclass, field
from typing import Sequence, Union, Callable, Generator, Dict, Optional, Any
from types import MethodType

from fastNLP.io import DataBundle

from fastie.envs import get_flag
from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils import Registry

NER = Registry('NER')
RE = Registry('RE')
EE = Registry('EE')


@dataclass
class BaseTaskConfig(BaseNodeConfig):
    cuda: bool = field(
        default=False,
        metadata=dict(
            help=
            'Whether to use your NVIDIA graphics card to accelerate the '
            'process.', existence=True))
    lr: float = field(default=2e-5,
                      metadata=dict(help='Learning rate during training.',
                                    existence='train'))
    batch_size: int = field(default=8,
                            metadata=dict(help='Batch size.', existence=True))
    load_model: Optional[str] = field(
        default=None,
        metadata=dict(help='Initialize the model with the saved parameters.',
                      existence=True))


class BaseTask(BaseNode):
    """所有任务的基类.

    Args:
        :cuda (bool)[train,evaluate,inference]=False: 是否使用显卡加速训练.
        :batch_size (int)[train]=32: batch size.
        :lr (float)[train]=2e-5: 学习率.
        :load_model (str)[train,evaluate,inference]=None: 加载模型的路径.
    """
    _config = BaseTaskConfig()

    def __init__(self,
                 cuda: Union[bool, int, Sequence[int]] = False,
                 **kwargs):
        BaseNode.__init__(self, **kwargs)
        self.cuda = cuda

        # if get_flag() is None:
        object.__setattr__(self, 'run', self.generate_run_func(self.run))

    def generate_run_func(self, run_func: Callable):

        def run(data_bundle: DataBundle):
            base_parameters: Dict[str, Union[int, str, Sequence]] = dict()
            if isinstance(self.cuda, bool):
                if self.cuda:
                    base_parameters['device'] = 0
                else:
                    base_parameters['device'] = 'cpu'
            elif isinstance(self.cuda, Sequence) and isinstance(
                    self.cuda[0], int):
                base_parameters['device'] = self.cuda

            if get_flag() is None:
                raise ValueError('You should set the flag first.')
            else:
                yield {**base_parameters, **run_func(data_bundle)}

        return run

    def run(self, data_bundle: DataBundle):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if isinstance(self.run, Generator):
            return self.run
        else:
            return self.run(*args, **kwargs)
