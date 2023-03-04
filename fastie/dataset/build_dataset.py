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
