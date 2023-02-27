from fastie.dataset.BaseDataset import DATASET
from fastie.dataset.io.sentence import Sentence
from fastie.envs import get_config, get_flag

from typing import Union, Optional, Sequence

from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle

def build_dataset(
        dataset: Optional[Union[str, Sequence[str],
        dict, Sequence[dict], DataSet, DataBundle]]) -> DataBundle:
    data_bundle = DataBundle()
    if dataset is None:
        if "dataset" not in get_config().keys():
            data_bundle = DATASET.get(get_config()["dataset"])()
    else:
        if isinstance(dataset, str) or isinstance(dataset, Sequence) \
                and isinstance(dataset[0], str):
            data_bundle = Sentence(dataset)()
        if isinstance(dataset, dict):
            dataset = [dataset]
        if isinstance(dataset, Sequence) and isinstance(dataset[0], dict):
            dataset = DataSet([Instance(**sample) for sample in dataset])
        if isinstance(dataset, DataSet):
            if get_flag() == "train":
                data_bundle = DataBundle(datasets={"train": dataset})
            elif get_flag() == "test":
                data_bundle = DataBundle(datasets={"test": dataset})
            elif get_flag() == "infer":
                data_bundle = DataBundle(datasets={"infer": dataset})
    return data_bundle
