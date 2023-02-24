from fastNLP.io import DataBundle

from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils import Registry
from fastie.envs import get_flag

from dataclasses import dataclass, field

from typing import Sequence, Union, Callable, Generator

NER = Registry("NER")
RE = Registry("RE")
EE = Registry("EE")

@dataclass
class BaseTaskConfig(BaseNodeConfig):
    cuda: bool = field(default=False,
                       metadata=dict(
                           help="Whether to use your NVIDIA graphics card to accelerate the process.",
                           existence=True
                       ))
    lr: float = field(default=2e-5,
                      metadata=dict(
                          help="Learning rate during training.",
                          existence="train"
                      ))
    batch_size: int = field(default=8,
                            metadata=dict(
                                help="Batch size.",
                                existence=True
                            ))
    load_model: str = field(default=None,
                            metadata=dict(
                                help="Initialize the model with the saved parameters.",
                                existence=True
                            ))


class BaseTask(BaseNode):
    """ 所有任务的基类

    Args:
        :cuda (bool)[train,evaluate,inference]=False: 是否使用显卡加速训练.
        :batch_size (int)[train]=32: batch size.
        :lr (float)[train]=2e-5: 学习率.
        :load_model (str)[train,evaluate,inference]=None: 加载模型的路径.

    """
    _config = BaseTaskConfig()
    def __init__(self,
                 cuda: Union[bool, int, Sequence[int]] = False,
                 batch_size: int = 32,
                 lr: float = 2e-5,
                 load_model: str = None,
                 **kwargs):
        BaseNode.__init__(self, **kwargs)
        self.cuda = cuda
        self.batch_size = batch_size
        self.lr = lr
        self.load_model = load_model

        # if get_flag() is None:
        self.run = self.generate_run_func(self.run)


    def generate_run_func(self, run_func: Callable):
        def run(data_bundle: DataBundle):
            base_parameters = dict()
            if isinstance(self.cuda, bool):
                if self.cuda:
                    base_parameters["device"] = 0
                else:
                    base_parameters["device"] = 'cpu'
            elif isinstance(self.cuda, Sequence) and isinstance(self.cuda[0], int):
                base_parameters["device"] = self.cuda

            if get_flag() is None:
                raise ValueError("You should set the flag first.")
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
