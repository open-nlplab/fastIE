import os
import sys
from functools import reduce
from typing import Union, Optional, Set, Tuple, List

from fastNLP import Vocabulary, Instance
from fastNLP.io import DataBundle

from fastie.envs import get_flag, CONFIG_FLAG, FASTIE_HOME, logger, set_task, \
    set_dataset
from fastie.utils.config import Config


def generate_tag_vocab(
        data_bundle: DataBundle,
        unknown: Optional[str] = 'O',
        base_mapping: Optional[dict] = None) -> Optional[Vocabulary]:
    """根据数据集中的已标注样本构建 tag_vocab.

    :param data_bundle: :class:`~fastNLP.io.DataBundle` 对象
    :param unknown: 未知标签的标记
    :param base_mapping: 基础映射，例如 ``{"label": 0}`` 或者 ``{0: "label"}``
        函数将在确保 ``base_mapping`` 中的标签不会被覆盖改变的前提下构造 vocab
    :return: 如果存在已标注样本，则返回构造成功的 :class:`~fastNLP.Vocabulary` 对象，
        否则返回空的 ``None``
    """
    if 'train' in data_bundle.datasets.keys() \
            or 'dev' in data_bundle.datasets.keys() \
            or 'test' in data_bundle.datasets.keys():
        # 存在已标注样本
        tag_vocab = Vocabulary(padding=None, unknown=unknown)

        def construct_vocab(instance: Instance):
            # 当然，用来 infer 的数据集是无法构建的，这里判断一下
            if 'entity_mentions' in instance.keys():
                for entity_mention in instance['entity_mentions']:
                    tag_vocab.add(entity_mention[1])
            return instance

        data_bundle.apply_more(construct_vocab)
        if base_mapping:
            base_word2idx = {}
            base_idx2word = {}
            if isinstance(base_mapping, dict) and isinstance(
                    list(base_mapping.keys())[0], str):
                base_word2idx = base_mapping
                base_idx2word = \
                    {word: idx for idx, word in base_word2idx.items()}
            elif isinstance(base_mapping, dict) and isinstance(
                    list(base_mapping.keys())[0], int):
                base_idx2word = base_mapping
                base_word2idx = \
                    {word: idx for idx, word in base_idx2word.items()}
            for key, value in tag_vocab.word2idx.items():
                if key not in base_word2idx.keys():
                    # 线性探测法
                    while value in base_idx2word.keys():
                        value += 1
                    base_word2idx[key] = value
                    base_idx2word[value] = key
            tag_vocab._word2idx = base_word2idx
            tag_vocab._idx2word = base_idx2word
        return tag_vocab
    else:
        # 无以标注样本
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
        * 为 ``-1`` 时
         表示出现冲突且无法矫正，请抛弃加载得到的 ``loaded_tag_vocab``
         使用返回的 ``tag_vocab``
        * 为 ``0`` 时
         无可用的 ``tag_vocab``，将直接输出 idx
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
        logger.warn('Error: No tag dictionary is available. ')
        return 0, None
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
                if set(word2idx.keys()) == set(  # type: ignore [union-attr]
                        tag_vocab.word2idx.keys()
                ):  # type: ignore [union-attr]
                    tag_vocab._word2idx.update(word2idx)
                    tag_vocab._idx2word.update(idx2word)
                    return 1, tag_vocab
                elif set(tag_vocab.word2idx.keys()  # type: ignore [union-attr]
                         ).issubset(set(tag_vocab.word2idx.keys())):
                    tag_vocab._word2idx.update(word2idx)
                    tag_vocab._idx2word.update(idx2word)
                    return 1, tag_vocab
                else:
                    logger.warn(
                        'The tag dictionary '  # type: ignore [union-attr]
                        f"`\n[{','.join(list(tag_vocab._word2idx.keys()))}]`\n"  # type: ignore [union-attr]
                        ' loaded from the model is not the same as the '
                        'tag dictionary '
                        f"\n`[{','.join(list(word2idx.keys()))}]`\n"
                        ' built from the dataset, so the loaded model may be '
                        'discarded')
                    return -1, tag_vocab
            else:
                return 1, tag_vocab
        else:
            tag_vocab._word2idx = word2idx
            tag_vocab._idx2word = idx2word
            return 1, tag_vocab
    return 0, None


def parse_config(_config: object) -> Optional[dict]:
    config = dict()
    if isinstance(_config, dict):
        for key, value in _config.items():
            if key == 'task':
                set_task(value)
            if key == 'dataset':
                set_dataset(value)
            if not key.startswith('_'):
                config[key] = value
        return config
    elif isinstance(_config, str):
        if os.path.exists(_config) and os.path.isfile(
                _config) and _config.endswith('.py'):
            if CONFIG_FLAG == 'dict':
                config_dict = reduce(lambda x, y: {
                    **x,
                    **y
                }, [
                    value
                    for value in Config.fromfile(_config)._cfg_dict.values()
                    if isinstance(value, dict)
                ])
                return parse_config(config_dict)
            elif CONFIG_FLAG == 'class':
                config_obj = Config.fromfile(_config)._cfg_dict.Config()
                config_dict = {
                    key: getattr(config_obj, key)
                    for key in dir(config_obj) if not key.startswith('_')
                }
                return parse_config(config_dict)
        else:
            for root, dirs, files in os.walk(
                    os.path.join(FASTIE_HOME, 'configs')):
                for file in files:
                    if _config == file.replace('.py', ''):
                        return parse_config(os.path.join(root, file))
        return None
    else:
        for key in _config.__dir__():
            if key == 'task':
                set_task(getattr(_config, key))
            if key == 'dataset':
                set_dataset(getattr(_config, key))
            if not key.startswith('_'):
                config[key] = getattr(_config, key)
        return config


def inspect_function_calling(func_name: str) -> Optional[Set[str]]:
    import inspect
    frame_info_list = inspect.stack()
    argument_user_provided = []
    for i in range(len(frame_info_list)):
        if frame_info_list[i + 1].function == func_name:
            for k in range(i, len(frame_info_list)):
                if frame_info_list[k + 1].function != func_name:
                    co_const = frame_info_list[k + 1].frame.f_code.co_consts
                    if len(co_const) > 1:
                        for j in range(len(co_const) - 1, -1, -1):
                            if isinstance(co_const[j], tuple) and \
                                    isinstance(co_const[j][0], str):
                                argument_user_provided.extend(co_const[j])
                                break
                    if 'args' in frame_info_list[k].frame.f_locals.keys():
                        argument_list = \
                            frame_info_list[k + 1].\
                        frame.f_locals[func_name].__code__.co_varnames
                        argument_user_provided.\
                            extend(argument_list[:len(
                            frame_info_list[k].frame.f_locals['args'])])
                    # 转换为 set 去除重复项
                    return set(argument_user_provided)
    return None

def inspect_metrics(parameters: dict = {}) -> List[str]:
    """
    根据参数中的 metrics 字段，返回真正检验结果中可能存在的 metric 名称.

    例如, 输入参数为 {'metrics': {'accuracy': Accuracy(), 'f1': SpanFPreRecMetric()}}，
    则返回 ['accuracy#acc', 'f1#f', 'f1#pre', 'f1#rec'].

    :param parameters: 可以用于 :class:`fastNLP.Trainer` 的参数字典.
    :return: 返回可能存在的 metric 名称列表.
    """
    from fastNLP import Trainer

    if 'metrics' not in parameters:
        return []
    try:
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        trainer = Trainer(**parameters)
        result = trainer.evaluator.run(num_eval_batch_per_dl=1)
        sys.stdout = stdout
        return list(result.keys())
    except Exception as e:
        logger.error(e)
        return []