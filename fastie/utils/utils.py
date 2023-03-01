import os
from functools import reduce
from typing import Union, Optional, Tuple

from fastNLP import Vocabulary, Instance
from fastNLP.io import DataBundle

from fastie.envs import get_flag, global_config, config_flag, FASTIE_HOME
from fastie.utils.config import Config


def generate_tag_vocab(data_bundle: DataBundle) -> Optional[Vocabulary]:
    """根据数据集中的已标注样本构建 tag_vocab.

    :param data_bundle: :class:`~fastNLP.io.DataBundle` 对象
    :return: 如果存在已标注样本，则返回构造成功的 :class:`~fastNLP.Vocabulary` 对象，
        否则返回空的 ``None``
    """
    if 'train' in data_bundle.datasets.keys() \
            or 'dev' in data_bundle.datasets.keys() \
            or 'test' in data_bundle.datasets.keys():
        # 存在已标注样本
        tag_vocab = Vocabulary()

        def construct_vocab(instance: Instance):
            # 当然，用来 infer 的数据集是无法构建的，这里判断一下
            if 'entity_mentions' in instance.keys():
                for entity_mention in instance['entity_mentions']:
                    tag_vocab.add(entity_mention[1])
            return instance

        data_bundle.apply_more(construct_vocab)
        return tag_vocab
    else:
        return None


def check_loaded_tag_vocab(
        loaded_tag_vocab: Optional[Union[dict, Vocabulary]],
        tag_vocab: Optional[Vocabulary]) -> Tuple[int, Optional[Vocabulary]]:
    """检查加载的 tag_vocab 是否与新生成的 tag_vocab 一致.

    :param loaded_tag_vocab: 从 ``checkpoint`` 中加载得到的 ``tag_vocab``;
        可以为 ``dict`` 类型，也可以是 :class:`~fastNLP.Vocabulary` 类型。
    :param tag_vocab: 从数据集中构建的 ``tag_vocab``;
    :return: 检查的结果信号和可使用的 ``tag_vocab``，信号取值：

        * 为 ``1`` 时
         表示一致或可矫正的错误，可以直接使用返回的 ``tag_vocab``
        * 为 ``0`` 时
         表示出现冲突且无法矫正，请抛弃加载得到的 ``loaded_tag_vocab``
         使用返回的 ``tag_vocab``
        * 为 ``-1`` 时
         无可用的 ``tag_vocab``，程序即将退出
    """
    idx2word = None
    word2idx = None
    if loaded_tag_vocab is not None:
        if isinstance(loaded_tag_vocab, Vocabulary):
            idx2word = loaded_tag_vocab.idx2word
            word2idx = loaded_tag_vocab.word2idx
        elif isinstance(list(loaded_tag_vocab.keys())[0], int):
            idx2word = loaded_tag_vocab
            word2idx = {word: idx for idx, word in idx2word.items()}
        elif isinstance(list(loaded_tag_vocab.keys())[0], str):
            word2idx = loaded_tag_vocab
            idx2word = {idx: word for word, idx in word2idx.items()}
    if loaded_tag_vocab is None and tag_vocab is None:
        print('Error: No tag dictionary is available. ')
        return -1, None
    if loaded_tag_vocab is None and tag_vocab is not None:
        return 1, tag_vocab
    if loaded_tag_vocab is not None and tag_vocab is None:
        tag_vocab = Vocabulary()
        tag_vocab._word2idx = word2idx
        tag_vocab._idx2word = idx2word
        return 1, tag_vocab
    if loaded_tag_vocab is not None and tag_vocab is not None:
        if get_flag() != 'infer':
            if word2idx != tag_vocab.word2idx:
                if set(word2idx.keys()) == set(tag_vocab.word2idx.keys()): # type: ignore [union-attr]
                    tag_vocab._word2idx.update(word2idx)
                    tag_vocab._idx2word.update(idx2word)
                    return 1, tag_vocab
                else:
                    print(
                        'Warning: The tag dictionary '
                        f"[{','.join(list(tag_vocab._word2idx.keys()))}]" # type: ignore [union-attr]
                        ' loaded from the model is not the same as the '
                        'tag dictionary '
                        f"[{','.join(list(word2idx.keys()))}]"
                        ' built from the dataset, so the loaded model may be '
                        'discarded')
                    return 0, tag_vocab
            else:
                return 1, tag_vocab
        else:
            tag_vocab._word2idx = word2idx
            tag_vocab._idx2word = idx2word
            return 1, tag_vocab


def set_config(_config: object) -> Optional[dict]:
    if isinstance(_config, dict):
        for key, value in _config.items():
            if not key.startswith('_'):
                global_config[key] = value
        return global_config
    elif isinstance(_config, str):
        if os.path.exists(_config) and os.path.isfile(
                _config) and _config.endswith('.py'):
            if config_flag == 'dict':
                config_dict = reduce(lambda x, y: {
                    **x,
                    **y
                }, [
                    value
                    for value in Config.fromfile(_config)._cfg_dict.values()
                    if isinstance(value, dict)
                ])
                return set_config(config_dict)
            elif config_flag == 'class':
                config_obj = Config.fromfile(_config)._cfg_dict.Config()
                config_dict = {
                    key: getattr(config_obj, key)
                    for key in dir(config_obj) if not key.startswith('_')
                }
                return set_config(config_dict)
        else:
            for root, dirs, files in os.walk(
                    os.path.join(FASTIE_HOME, 'configs')):
                for file in files:
                    if _config == file.replace('.py', ''):
                        return set_config(os.path.join(root, file))
        return None
    else:
        for key in _config.__dir__():
            if not key.startswith('_'):
                global_config[key] = getattr(_config, key)
        return global_config
