from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils import Registry
from fastie.dataset import Sentence
from fastie.tasks import build_task

from fastNLP import DataSet
from fastNLP.io import DataBundle

from typing import Union, Sequence, Generator, Callable

CONTROLLER = Registry("CONTROLLER")


class BaseController(BaseNode):
    def __init__(self):
        BaseNode.__init__(self)

    def run(self,
            parameters_or_data: Union[dict, DataBundle, DataSet, str, Sequence[str]] = dict()):
        if isinstance(parameters_or_data, Generator):
            parameters_or_data = next(parameters_or_data)
        if isinstance(parameters_or_data, Callable):
            parameters_or_data = parameters_or_data()
        if isinstance(parameters_or_data, dict):
            return parameters_or_data
        else:
            if isinstance(parameters_or_data, DataBundle):
                data_bundle = parameters_or_data
            if isinstance(parameters_or_data, DataSet):
                data_bundle = DataBundle(datasets={"generate": parameters_or_data})
            elif isinstance(parameters_or_data, str) or isinstance(parameters_or_data, Sequence):
                data_bundle = Sentence(parameters_or_data)()
            return build_task()(data_bundle)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

