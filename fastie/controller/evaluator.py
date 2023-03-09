"""Evaluator for FastIE."""
__all__ = ['Evaluator', 'EvaluatorConfig']
from fastie.controller.BaseController import BaseController, CONTROLLER
from fastie.envs import set_flag, logger
from fastie.node import BaseNodeConfig

from fastNLP import DataSet
from fastNLP.io import DataBundle
from fastNLP import Evaluator as FastNLP_Evaluator

from typing import Union, Sequence, Optional

from dataclasses import dataclass


@dataclass
class EvaluatorConfig(BaseNodeConfig):
    """验证器的配置类."""
    pass


@CONTROLLER.register_module('evaluator')
class Evaluator(BaseController):
    """验证器 用于对任务在 ``test`` 数据集上进行检验，并输出 ``test`` 数据集上的 ``metric``"""

    def __init__(self):
        super(Evaluator, self).__init__()
        set_flag('eval')

    def run(
        self,
        parameters_or_data: Optional[Union[dict, DataBundle, DataSet]] = None
    ) -> dict:
        """验证器的 ``run`` 方法，用于实际地对传入的 ``task`` 或是数据集进行验证.

        也可以使用命令行模式, 例如:

        .. code-block:: console
            :linenos:
            $ fastie-inference --task ner/bert --dataset conll2003 --save_path result.jsonl

        :param parameters_or_data: 既可以是 task，也可以是数据集:
            * 为 ``task`` 时, 应为 :class:`~fastie.BaseTask` 对象 ``run``
            方法的返回值, 例如:
                >>> from fastie.tasks import BertNER
                >>> task = BertNER().run()
                >>> Evaluator().run(task)
            * 为数据集，可以是 ``[dict, DataSet, DataBundle, None]`` 类型的数据集：
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
                * ``DataBundle`` 类型的数据集，必须包含 ``test`` 子集, 例如:
                    >>> from fastNLP import DataSet, Instance
                    >>> from fastNLP.io import DataBundle
                    >>> dataset = DataBundle(datasets={'test': DataSet([Instance(tokens=[ "It", "is", "located", "in", "Seoul", "." ],
                    >>>                                                          entity_motions=([4], "LOC"))])})
                * ``None`` 会自动寻找 ``config`` 中的 ``dataset``, 例如:
                    >>> config = {'dataset': 'conll2003'}
                    >>> Evaluator.from_config(config).run()

        :return: ``dict`` 类型的 ``metric`` 结果, 例如:
            >>> {'acc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        """
        parameters_or_data = BaseController.run(self, parameters_or_data)
        if parameters_or_data is None:
            logger.error(
                'Evaluating tool do not allow task and dataset to be left '
                'empty. ')
            exit(1)
        evaluator = FastNLP_Evaluator(**parameters_or_data)
        return evaluator.run()
