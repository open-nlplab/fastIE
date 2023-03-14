"""控制器基类."""

__all__ = ['BaseController', 'CONTROLLER']

from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils import Registry, parse_config
from fastie.tasks import build_task
from fastie.dataset.build_dataset import build_dataset

from fastNLP import DataSet
from fastNLP.io import DataBundle

from typing import Union, Sequence, Generator, Optional

CONTROLLER = Registry('CONTROLLER')


class BaseController(BaseNode):
    """Base class for all controllers."""

    def __init__(self, **kwargs):
        BaseNode.__init__(self, **kwargs)

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str],
                                               Sequence[dict]]] = None):
        """控制器基类的 ``run`` 方法，用于实际地对传入的 ``task`` 或是数据集进行训练, 验证或推理.

        :param parameters_or_data: 既可以是 task，也可以是数据集:
            * 为 ``task`` 时, 应为 :class:`~fastie.BaseTask` 对象 ``run``
            方法的返回值, 例如:
                >>> from fastie.tasks import BertNER
                >>> task = BertNER().run()
                >>> Trainer().run(task)
            * 为数据集，可以是 ``[dict, DataSet, DataBundle, str,
            Sequence[str], Sequence[dict], None]`` 类型的数据集：
                * ``dict`` 类型的数据集，例如:
                    >>> dataset = {'tokens': [ "It", "is", "located", "in", "Seoul", "." ],
                    >>>            'entity_motions': [([4], "LOC")]}
                * ``Sequence[dict]`` 类型的数据集，例如:
                    >>> dataset = [{'tokens': [ "It", "is", "located", "in", "Seoul", "." ],
                    >>>            'entity_motions': [([4], "LOC")]}]
                * ``DataSet`` 类型的数据集，例如:
                    >>> from fastNLP import DataSet, Instance
                    >>> dataset = DataSet([Instance(tokens=[ "It", "is", "located", "in", "Seoul", "." ],
                    >>>                              entity_motions=([4], "LOC"))])
                * ``DataBundle`` 类型的数据集，例如:
                    >>> from fastNLP import DataSet, Instance
                    >>> from fastNLP.io import DataBundle
                    >>> dataset = DataBundle(datasets={'train': DataSet([Instance(tokens=[ "It", "is", "located", "in", "Seoul", "." ],
                    >>>                                                           entity_motions=([4], "LOC"))])})
                * ``str`` 类型的数据集，会自动根据空格分割转换为 ``token``, 仅适用于推理，
                详见 :class:`fastie.dataset.io.sentence.Sentence` 例如:
                    >>> dataset = "It is located in Seoul ."
                * ``Sequence[str]`` 类型的数据集，会自动根据空格分割转换为 ``token``, 仅适用于推理,
                详见 :class:`fastie.dataset.io.sentence.Sentence`, 例如:
                    >>> dataset = ["It is located in Seoul .", "It is located in Beijing ."]
                * ``None`` 会自动寻找 ``config`` 中的 ``dataset``, 例如:
                    >>> config = {'dataset': 'conll2003'}
                    >>> Trainer.from_config(config).run()

        :return: 训练, 验证, 或推理的结果:
            * 训练时, 返回任务的 ``state_dict``
            * 验证时, 返回验证集的 ``metric``
            * 推理时, 返回推理结果
        """
        if callable(parameters_or_data):
            parameters_or_data = parameters_or_data()
        if isinstance(parameters_or_data, Generator):
            parameters_or_data = next(parameters_or_data)
        if isinstance(parameters_or_data, dict) \
                and 'model' in parameters_or_data.keys():
            return parameters_or_data
        else:
            # 下面的是直接传入数据集的情况，需要根据 global_config 构建 task
            data_bundle = build_dataset(parameters_or_data,
                                        dataset_config=self._overload_config)
            parameters_or_data = build_task(self._overload_config)(data_bundle)
            if isinstance(parameters_or_data, Generator):
                parameters_or_data = next(parameters_or_data)
            return parameters_or_data

    def __call__(self, *args, **kwargs):
        """重载 ``__call__`` 方法，使得控制器可以直接调用 ``run`` 方法.

        :param args: 与 ``run`` 方法的参数一致
        :param kwargs: 与 ``run`` 方法的参数一致
        :return: 与 ``run`` 方法的返回结果一致
        """
        return self.run(*args, **kwargs)
