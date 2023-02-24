from dataclasses import dataclass, field

from datasets import load_dataset
from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.io import DataBundle

from fastie.envs import get_flag
from .. import DATASET, BaseDataset, BaseDatasetConfig


@dataclass
class WikiannLoaderConfig(BaseDatasetConfig):
    language: str = field(
        default='zh',
        metadata=dict(
            help=
            'Select which language subset in wikiann. Refer to https://huggingface.co/datasets/wikiann .',
            existence=True))


@DATASET.register_module('wikiann')
class WikiannLoader(BaseDataset):
    """Wikiann."""
    _config = WikiannLoaderConfig()

    def __init__(self, language: str = 'zh', **kwargs):
        BaseDataset.__init__(self, **kwargs)
        self.language = language

    def run(self):
        if self.language == 'zh':
            raw_dataset = load_dataset('wikiann', 'zh')
        datasets = {}
        if get_flag() == 'train' or get_flag() == 'evaluation':
            datasets['train'] = DataSet([
                Instance(tokens=sample['tokens'], ner_tags=sample['ner_tags'])
                for sample in raw_dataset['train']
            ])
            datasets['evaluate'] = DataSet([
                Instance(tokens=sample['tokens'], ner_tags=sample['ner_tags'])
                for sample in raw_dataset['validation']
            ])
        data_bundle = DataBundle(datasets=datasets)
        word2idx = {
            'O': 0,
            'B-PER': 1,
            'I-PER': 2,
            'B-ORG': 3,
            'I-ORG': 4,
            'B-LOC': 5,
            'I-LOC': 6
        }
        idx2word = {value: key for key, value in word2idx.items()}
        ner_vocab = Vocabulary()
        ner_vocab._word2idx = word2idx
        ner_vocab._idx2word = idx2word
        data_bundle.set_vocab(ner_vocab, 'ner')
        return data_bundle
