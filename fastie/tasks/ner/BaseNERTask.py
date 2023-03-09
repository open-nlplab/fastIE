"""Base class for NER tasks."""
__all__ = ['BaseNERTask', 'BaseNERTaskConfig']
from fastie.tasks.BaseTask import BaseTask, BaseTaskConfig
from fastie.utils.utils import generate_tag_vocab, check_loaded_tag_vocab
from fastie.envs import logger

from fastNLP import DataSet, Vocabulary
from fastNLP.io import DataBundle

import abc

from typing import Union, Sequence, Optional


class BaseNERTaskConfig(BaseTaskConfig):
    """NER 任务所需参数."""
    pass


class BaseNERTask(BaseTask, metaclass=abc.ABCMeta):
    """FastIE NER 任务基类.

    :param load_model: 模型文件的路径或者模型名
    :param save_model_folder: ``topk`` 或 ``load_best_model`` 保存模型的文件夹
    :param batch_size: batch size
    :param epochs: 训练的轮数
    :param monitor: 根据哪个 ``metric`` 选择  ``topk`` 和 ``load_best_model``；
        如果不设置，则默认使用结果中的第一个 ``metric``
    :param is_large_better: ``metric`` 中 ``monitor`` 监控的指标是否越大越好
    :param topk: 将 ``metric`` 中 ``monitor`` 监控的指标最好的 k 个模型保存到
        ``save_model_folder`` 中
    :param load_best_model: 是否在训练结束后将 ``metric`` 中 ``monitor`` 监控的指标最
        好的模型保存到 ``save_model_folder`` 中，并自动加载到 ``task`` 中
    :param fp16: 是否使用混合精度训练
    :param evaluate_every: 训练过程中检验的频率,
        ``topk`` 和 ``load_best_model`` 将在所有的检验中选择:
            * 为 ``0`` 时则训练过程中不进行检验
            * 如果为正数，则每 ``evaluate_every`` 个 batch 进行一次检验
            * 如果为负数，则每 ``evaluate_every`` 个 epoch 进行一次检验
    :param device: 指定具体训练时使用的设备
        device 的可选输入如下所示:
            * *str*: 例如 ``'cpu'``, ``'cuda'``, ``'cuda:0'``, ``'cuda:1'``,
            `'gpu:0'`` 等；
            * *int*: 将使用 ``device_id`` 为该值的 ``gpu`` 进行训练；如果值为 -1，那么
            默认使用全部的显卡；
            * *list(int)*: 如果多于 1 个device，应当通过该种方式进行设定，将会启用分布式程序。
    """

    _config = BaseNERTaskConfig()
    _help = 'Base class for NER tasks. '

    def __init__(self,
                 load_model: str = '',
                 save_model_folder: str = '',
                 batch_size: int = 32,
                 epochs: int = 20,
                 monitor: str = '',
                 is_large_better: bool = True,
                 topk: int = 0,
                 load_best_model: bool = False,
                 fp16: bool = False,
                 evaluate_every: int = -1,
                 device: Union[int, Sequence[int], str] = 'cpu',
                 **kwargs):
        super().__init__(load_model, save_model_folder, batch_size, epochs,
                         monitor, is_large_better, topk, load_best_model, fp16,
                         evaluate_every, device, **kwargs)

    def on_generate_and_check_tag_vocab(self,
                                        data_bundle: DataBundle,
                                        state_dict: Optional[dict]) \
            -> Optional[Vocabulary]:
        """根据数据集中每个样本 `sample['entity_motions'][i][1]` 生成标签词典。 如果加载模型得到的
        ``state_dict`` 中存在 ``tag_vocab``，则检查是否与根据 ``data_bundle`` 生成的 tag_vocab
        一致 (优先使用加载得到的 tag_vocab)。

        :param data_bundle: 原始数据集，
            可能包含 ``train``、``dev``、``test``、``infer`` 四种，需要分类处理。
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 标签词典，可能为 ``None``
        """
        tag_vocab = None
        if state_dict is not None and 'tag_vocab' in state_dict:
            tag_vocab = state_dict['tag_vocab']
        signal, tag_vocab = check_loaded_tag_vocab(
            tag_vocab, generate_tag_vocab(data_bundle))
        if signal == -1:
            logger.warning(f'It is detected that the model label vocabulary '
                           f'conflicts with the dataset label vocabulary, '
                           f'so the model loading may fail. ')
        return tag_vocab
