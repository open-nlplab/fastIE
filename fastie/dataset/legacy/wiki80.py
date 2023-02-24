import json

import requests
from fastNLP import DataSet as FastNLP_Dataset, Instance, Vocabulary

from fastie.dataset import BaseDataset, DATASET


@DATASET.register_module("wiki80")
class Dataset(BaseDataset):
    """ WikiANN

    """
    def __init__(self):
        super(Dataset).__init__()

    @property
    def dataset(self):
        rel2id = json.loads(requests.get(
            r"https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki80/wiki80_rel2id.json").text)
        relation_vocab = Vocabulary(unknown=None, padding=None)
        relation_vocab.word2idx = rel2id
        relation_vocab.idx2word = {value: key for key, value in rel2id.items()}
        train_text = requests.get(
            r"https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki80/wiki80_train.txt")
        train_dataset = FastNLP_Dataset()
        for line in train_text.split('\n'):
            sample = json.loads(line)
            train_dataset.append(Instance(token=sample["token"],
                                          h=sample["h"]["pos"],
                                          t=sample["t"]["pos"],
                                          tag=sample["relation"]))
