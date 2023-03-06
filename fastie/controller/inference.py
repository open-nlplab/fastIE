"""
推理器
"""
__all__ = [
    'Inference',
    'InferenceConfig',
    'InferenceMetric'
]
import json
from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from typing import Union, Sequence, Optional

from fastNLP import Evaluator, DataSet, Metric
from fastNLP.io import DataBundle

from fastie.controller.BaseController import BaseController, CONTROLLER
from fastie.envs import set_flag, logger
from fastie.node import BaseNodeConfig


class InferenceMetric(Metric):
    """
    用于保存推理结果的 Metric
    :param save_path: 保存路径, 应为一个文件名, 例如 ``result.jsonl``
    :param verbose: 是否打印推理结果
    """

    def __init__(self, save_path: Optional[str] = None, verbose: bool = True):
        super().__init__(aggregate_when_get_metric=False)
        self.result: list = []
        self.save_path: Optional[str] = save_path
        self.verbose: bool = verbose

    def update(self, pred: Sequence[dict]):
        if self.save_path is not None:
            if self.backend.is_distributed() and self.backend.is_global_zero():
                with open(self.save_path, 'a+') as file:
                    file.write('\n'.join(
                        map(
                            lambda x: json.dumps(x),
                            reduce(lambda x, y: [*x, *y],
                                   self.all_gather_object(pred)))) + '\n')
            elif not self.backend.is_distributed():
                with open(self.save_path, 'a+') as file:
                    file.write('\n'.join(map(lambda x: json.dumps(x), pred)) +
                               '\n')
        if self.verbose and not self.backend.is_distributed():
            for sample in pred:
                # 判断一下不同的格式
                # 首先是 NER 小组约定的格式
                if 'entity_mentions' in sample.keys():
                    print('tokens: ', ' '.join(sample['tokens']))
                    print(
                        'pred:   ', ' '.join([
                            sample['tokens'][i] if i
                            in sample['entity_mentions'][0][0] else ''.join(
                                [' ' for j in range(len(sample['tokens'][i]))])
                            for i in range(len(sample['tokens']))
                        ]), f"  {sample['entity_mentions'][0][1]} -> "
                        f"{sample['entity_mentions'][0][2]}"
                        if len(sample['entity_mentions'][0]) == 3 else
                        f"  {sample['entity_mentions'][0][1]}")
                    if len(sample['entity_mentions']) > 1:
                        for entity_mention in sample['entity_mentions'][1:]:
                            print(
                                '        ', ' '.join([
                                    sample['tokens'][i]
                                    if i in entity_mention[0] else ''.join([
                                        ' ' for j in range(
                                            len(sample['tokens'][i]))
                                    ]) for i in range(len(sample['tokens']))
                                ]), f'  {entity_mention[1]} -> '
                                f'{entity_mention[2]}' if len(entity_mention)
                                == 3 else f'  {entity_mention[1]}')
                else:
                    # TODO: 其他类型的格式，例如为关系抽取小组制定的格式
                    pass
        self.result.extend(pred)

    def get_metric(self):
        return reduce(lambda x, y: x.extend(y),
                      self.all_gather_object(self.result))


def generate_step_fn(evaluator, batch):
    outputs = evaluator.evaluate_step(batch)
    content = '\n'.join(outputs['pred'])
    if getattr(evaluator, 'generate_save_path', None) is not None:
        with open(evaluator.generate_save_path, 'a+') as f:
            f.write(f'{content}\n')
    else:
        evaluator.result.extend(outputs['pred'])


@dataclass
class InferenceConfig(BaseNodeConfig):
    """
    推理器的配置
    """
    save_path: Optional[str] = field(
        default=None,
        metadata=dict(
            help='The path to save the generated results. If not set, output to '
            'the returned variable. ',
            existence=['infer']))
    verbose: bool = field(
        default=True,
        metadata=dict(
            help=
            'Whether to output the contents of each inference. Multiple cards '
            'are not supported. ',
            existence=['infer']))


@CONTROLLER.register_module('inference')
class Inference(BaseController):
    """
    推理器
    用于对任务在 ``infer`` 数据集上进行检验，并输出 ``infer`` 数据集上的推理结果

    也可以使用命令行模式, 例如:

        .. code-block:: console
            :linenos:
            $ fastie-infer --task ner/bert --dataset sentence --sentence It is located in Beijing --verbose

    :param save_path: 推理结果的保存路径, 应为一个文件名, 例如 ``result.jsonl``
    推理结果将保存到 ``save_path`` 中, 保存的格式为 ``jsonl`` 格式, 每行为一个样本的推理结果的 ``json`` 字符串

    :param verbose: 是否在推理的过程中实时打印推理结果
    """
    _config = InferenceConfig()
    _help = 'Inference tool for FastIE. '
    def __init__(self,
                 save_path: Optional[str] = None,
                 verbose: bool = True,
                 **kwargs):
        super(Inference, self).__init__(**kwargs)
        self.save_path: Optional[str] = save_path
        self.verbose: bool = verbose
        set_flag('infer')

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,\
                    Sequence[str]]] = None) -> Sequence[dict]:
        """
        验证器的 ``run`` 方法，用于实际地对传入的 ``task`` 或是数据集进行推理

        :param parameters_or_data: 既可以是 task，也可以是数据集:
            * 为 ``task`` 时, 应为 :class:`~fastie.BaseTask` 对象 ``run``
            方法的返回值, 例如:
                >>> from fastie.tasks import BertNER
                >>> task = BertNER().run()
                >>> Inference().run(task)
            * 为数据集，可以是 ``[dict, DataSet, DataBundle, None]`` 类型的数据集：
                * ``dict`` 类型的数据集，例如:
                    >>> dataset = {'tokens': [ "It", "is", "located", "in", "Seoul", "." ]}
                * ``Sequence[dict]`` 类型的数据集，例如:
                    >>> dataset = [{'tokens': [ "It", "is", "located", "in", "Seoul", "." ]}]
                * ``DataSet`` 类型的数据集，例如:
                    >>> from fastNLP import DataSet, Instance
                    >>> dataset = DataSet([Instance(tokens=[ "It", "is", "located", "in", "Seoul", "." ])])
                * ``DataBundle`` 类型的数据集，必须包含 ``infer`` 子集, 例如:
                    >>> from fastNLP import DataSet, Instance
                    >>> from fastNLP.io import DataBundle
                    >>> dataset = DataBundle(datasets={'infer': DataSet([Instance(tokens=[ "It", "is", "located", "in", "Seoul", "." ])])})
                * ``str`` 类型的数据集，会自动根据空格分割转换为 ``token``,
                详见 :class:`fastie.dataset.io.sentence.Sentence` 例如:
                    >>> dataset = "It is located in Seoul ."
                * ``Sequence[str]`` 类型的数据集，会自动根据空格分割转换为 ``token``,
                详见 :class:`fastie.dataset.io.sentence.Sentence`, 例如:
                    >>> dataset = ["It is located in Seoul .", "It is located in Beijing ."]
                * ``None`` 会自动寻找 ``config`` 中的 ``dataset``, 例如:
                    >>> config = {'dataset': 'conll2003'}
                    >>> Trainer.from_config(config).run()

            :return: ``List[dict]`` 类型的推理结果, 例如:
                >>> [{'tokens': [ "It", "is", "located", "in", "Seoul", "." ],
                >>>   'entity_motions': [([4], "LOC")]}]
            """
        parameters_or_data = BaseController.run(self, parameters_or_data)
        if parameters_or_data is None:
            logger.error(
                'Inference tool do not allow task and dataset to be left '
                'empty. ')
            exit(1)
        parameters_or_data['evaluate_fn'] = 'inference_step'
        parameters_or_data['verbose'] = False
        inference_metric = InferenceMetric(save_path=self.save_path,
                                           verbose=self.verbose)
        parameters_or_data['metrics'] = {'infer': inference_metric}
        evaluator = Evaluator(**parameters_or_data)
        evaluator.run()
        return inference_metric.result
