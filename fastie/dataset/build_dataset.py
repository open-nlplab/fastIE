"""Build dataset from different sources."""
__all__ = ['build_dataset']

from fastie.dataset.BaseDataset import DATASET
from fastie.dataset.io.sentence import Sentence
from fastie.envs import get_flag, get_dataset
from fastie.utils.utils import parse_config

from typing import Union, Optional, Sequence

from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle


def build_dataset(dataset: Optional[Union[str, Sequence[str], dict,
                                          Sequence[dict], DataSet,
                                          DataBundle]],
                  dataset_config: Optional[dict] = None) -> DataBundle:
    """从不同的来源构造数据集.

    :param dataset: 可以是 ``str`` 或 ``Sequence[str]`` 或 ``dict``
        或 ``Sequence[dict]`` 或 ``DataSet`` 或 ``DataBundle``:

            * 为 ``str`` 时, 将自动构建 ``Sentence`` 数据集, 该数据集只有一个 ``tokens`` 字段, 请用空格分割不同的 ``token``；
            * 为 ``Sequence[str]`` 时, 将自动构建 ``Sentence`` 数据集, 包含多个样本；
            * 为 ``dict`` 时, 将自动构建 ``DataSet`` 数据集, 键名将被映射到 ``DataSet`` 的 ``field_name``；
            * 为 ``Sequence[dict]`` 时, 将自动构建 ``DataSet`` 数据集, 包含多个样本；
            * 为 ``DataSet`` 时, 将自动构建 ``DataBundle`` 数据集, 并根据当前的 ``flag`` 自动决定 ``split`` 的名称, 例如 ``train`` ``dev`` ``test`` ``infer`` ；
            * 为 ``DataBundle`` 时, 直接返回该数据集；
            * 为 ``None`` 时, 根据配置文件中的 ``dataset`` 构建数据集.

    :param dataset_config: ``dataset`` 对象的参数
    :return: ``DataBundle`` 数据集
    """
    data_bundle = DataBundle()
    if dataset is None:
        if not get_dataset():
            raise ValueError('The dataset you want to use is not specified.')
        else:
            if dataset_config is None:
                data_bundle = DATASET.get(get_dataset())().run()
            else:
                data_bundle = DATASET.get(
                    get_dataset())(**parse_config(dataset_config)).run()
    else:
        if isinstance(dataset, str) or isinstance(dataset, Sequence) \
                and isinstance(dataset[0], str):
            data_bundle = Sentence(dataset)()  # type: ignore [arg-type]
        if isinstance(dataset, dict):
            dataset = [dataset]
        if isinstance(dataset, Sequence) and isinstance(dataset[0], dict):
            dataset = DataSet([Instance(**sample) for sample in dataset])
        if isinstance(dataset, DataSet):
            if get_flag() == 'train':
                data_bundle = DataBundle(datasets={'train': dataset})
            elif get_flag() == 'eval':
                data_bundle = DataBundle(datasets={'test': dataset})
            elif get_flag() == 'infer' or get_flag() == 'interact':
                data_bundle = DataBundle(datasets={'infer': dataset})
    return data_bundle
