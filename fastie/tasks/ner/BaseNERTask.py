"""Base class for NER tasks."""
__all__ = ['BaseNERTask', 'BaseNERTaskConfig']

import abc
from typing import Optional, Dict

from fastNLP import Vocabulary
from fastNLP.io import DataBundle

from fastie.envs import logger
from fastie.tasks.base_task import BaseTask, BaseTaskConfig
from fastie.utils.utils import generate_tag_vocab, check_loaded_tag_vocab


class BaseNERTaskConfig(BaseTaskConfig):
    """NER 任务所需参数."""
    pass


class BaseNERTask(BaseTask, metaclass=abc.ABCMeta):
    """FastIE NER 任务基类."""

    _config = BaseNERTaskConfig()
    _help = 'Base class for NER tasks. '

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_generate_and_check_tag_vocab(self,
                                        data_bundle: DataBundle,
                                        state_dict: Optional[dict]) \
            -> Dict[str, Vocabulary]:
        """根据数据集中每个样本 `sample['entity_motions'][i][1]` 生成标签词典。 如果加载模型得到的
        ``state_dict`` 中存在 ``tag_vocab``，则检查是否与根据 ``data_bundle`` 生成的 tag_vocab
        一致 (优先使用加载得到的 tag_vocab)。

        :param data_bundle: 原始数据集，
            可能包含 ``train``、``dev``、``test``、``infer`` 四种，需要分类处理。
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 标签词典，可能为 ``None``
        """
        tag_vocab = {}
        if state_dict is not None and 'tag_vocab' in state_dict:
            tag_vocab = state_dict['tag_vocab']
        generated_tag_vocab = generate_tag_vocab(data_bundle)
        for key, value in tag_vocab.items():
            if key not in generated_tag_vocab.keys():
                generated_tag_vocab[key] = check_loaded_tag_vocab(value,
                                                                  None)[1]
            else:
                signal, generated_tag_vocab[key] = check_loaded_tag_vocab(
                    value, generated_tag_vocab[key])
                if signal == -1:
                    logger.warning(
                        f'It is detected that the loaded ``{key}`` vocabulary '
                        f'conflicts with the generated ``{key}`` vocabulary, '
                        f'so the model loading may fail. ')
        return generated_tag_vocab
