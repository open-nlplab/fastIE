"""多阶段任务的基类."""
__all__ = ['SequentialTaskConfig', 'SequentialTask']
from fastie.tasks.base_task import BaseTask
from fastie.node import BaseNodeConfig, BaseNode
from fastie.envs import get_flag

from fastNLP.io import DataBundle

from dataclasses import dataclass
from typing import List
from abc import ABCMeta, abstractmethod


@dataclass
class SequentialTaskConfig(BaseNodeConfig):
    """多阶段任务的配置类."""
    pass


class SequentialTask(BaseNode, metaclass=ABCMeta):

    def __init__(self):
        self._tasks: List[BaseTask] = []

    @abstractmethod
    def on_train(self, data_bundle: DataBundle):
        """
        训练多阶段任务的逻辑
        :return:
        """

    @abstractmethod
    def on_eval(self, data_bundle: DataBundle):
        """
        验证多阶段任务的逻辑
        :return:
        """

    @abstractmethod
    def on_infer(self, data_bundle: DataBundle):
        """
        推理多阶段任务的逻辑
        :return:
        """

    def run(self, data_bundle: DataBundle):
        if get_flag() == 'train':
            return self.on_train(data_bundle)
        elif get_flag() == 'eval':
            return self.on_eval(data_bundle)
        elif get_flag() == 'infer':
            return self.on_infer(data_bundle)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
