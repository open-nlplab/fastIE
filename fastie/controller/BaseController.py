from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils import Registry
from fastie.dataset import Sentence
from fastie.tasks import build_task
from fastie.envs import get_config

from fastNLP import DataSet
from fastNLP.io import DataBundle

from typing import Union, Sequence, Generator, Optional

CONTROLLER = Registry('CONTROLLER')


class BaseController(BaseNode):

    def __init__(self, **kwargs):
        BaseNode.__init__(self, **kwargs)

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str]]] = dict()):
        if parameters_or_data is None:
            return None
        if isinstance(parameters_or_data, Generator):
            parameters_or_data = next(parameters_or_data)
        if callable(parameters_or_data):
            parameters_or_data = parameters_or_data()
        if isinstance(parameters_or_data, dict):
            return parameters_or_data
        else:
            if isinstance(parameters_or_data, DataBundle):
                data_bundle = parameters_or_data
            if isinstance(parameters_or_data, DataSet):
                if self.__class__.__name__ == 'Trainer':
                    data_bundle = DataBundle(
                        datasets={'train': parameters_or_data})
                elif self.__class__.__name__ == 'Evaluator':
                    data_bundle = DataBundle(
                        datasets={'valid': parameters_or_data})
                elif self.__class__.__name__ == 'Inference' \
                        or self.__class__.__name__ == 'Interactor':
                    data_bundle = DataBundle(
                        datasets={'infer': parameters_or_data})
            elif isinstance(parameters_or_data, str) or isinstance(
                    parameters_or_data, Sequence) \
                    and isinstance(parameters_or_data[0], str):
                data_bundle = Sentence(parameters_or_data)()
                parameters_or_data = build_task()(data_bundle)
                if isinstance(parameters_or_data, Generator):
                    parameters_or_data = next(parameters_or_data)
            return parameters_or_data

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
