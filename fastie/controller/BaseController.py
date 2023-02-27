from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils import Registry
from fastie.tasks import build_task
from fastie.dataset.build_dataset import build_dataset

from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle

from typing import Union, Sequence, Generator, Optional

CONTROLLER = Registry('CONTROLLER')


class BaseController(BaseNode):

    def __init__(self, **kwargs):
        BaseNode.__init__(self, **kwargs)

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str]]] = None):
        if isinstance(parameters_or_data, Generator):
            parameters_or_data = next(parameters_or_data)
        if callable(parameters_or_data):
            parameters_or_data = parameters_or_data()
        if isinstance(parameters_or_data, dict) \
                and "model" in parameters_or_data.keys():
            return parameters_or_data
        else:
            # 下面的是直接传入数据集的情况，需要根据 global_config 构建 task
            data_bundle = build_dataset(parameters_or_data)
            parameters_or_data = build_task()(data_bundle)
            if isinstance(parameters_or_data, Generator):
                parameters_or_data = next(parameters_or_data)
            return parameters_or_data

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
