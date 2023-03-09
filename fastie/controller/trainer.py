"""Trainer for FastIE."""
__all__ = ['Trainer', 'TrainerConfig']
from fastie.controller.BaseController import BaseController, CONTROLLER
from fastie.envs import set_flag, logger
from fastie.node import BaseNodeConfig

from fastNLP import DataSet
from fastNLP.io import DataBundle
from fastNLP import Trainer as FastNLP_Trainer

from typing import Union, Sequence, Optional

from dataclasses import dataclass, field


@dataclass
class TrainerConfig(BaseNodeConfig):
    """训练器的配置类."""
    pass


@CONTROLLER.register_module('trainer')
class Trainer(BaseController):
    """训练器 用于对任务在 ``train`` 数据集上进行训练，并输出 ``dev`` 数据集上的 ``metric``

    也可以使用命令行模式, 例如:

        .. code-block:: console
            :linenos:
            $ fastie-train --task ner/bert --dataset conll2003 --topk 3 --save_model model.pkl
    """
    _config = TrainerConfig()
    _help = 'Trainer for FastIE '

    def __init__(self):
        super(Trainer, self).__init__()
        set_flag('train')

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str]]] = None):
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
            * 为数据集，可以是 ``[dict, Sequence[dict], DataSet, DataBundle, None]`` 类型的数据集：
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
                * ``DataBundle`` 类型的数据集，必须包含 ``train`` 子集,
                如果有 ``dev`` 子集的话，则在每个 epoch 结束后进行检验,例如:
                    >>> from fastNLP import DataSet, Instance
                            >>> from fastNLP.io import DataBundle
                    >>> dataset = DataBundle(datasets={'train': DataSet([Instance(tokens=[ "It", "is", "located", "in", "Seoul", "." ],
                    >>>                                                          entity_motions=([4], "LOC"))])})
                * ``None`` 会自动寻找 ``config`` 中的 ``dataset``, 例如:
                    >>> config = {'dataset': 'conll2003'}
                    >>> Evaluator.from_config(config).run()

        :return: ``None``
        """
        parameters_or_data = BaseController.run(self, parameters_or_data)
        if parameters_or_data is None:
            logger.error(
                'Training tool do not allow task and dataset to be left '
                'empty. ')
            exit(1)
        trainer = FastNLP_Trainer(**parameters_or_data)
        trainer.run()
