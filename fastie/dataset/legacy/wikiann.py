"""这个类还没写好，请勿参考."""

from dataclasses import dataclass, field

from datasets import load_dataset
from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle

import numpy as np

from fastie.dataset.BaseDataset import DATASET, BaseDataset, BaseDatasetConfig


@dataclass
class WikiannConfig(BaseDatasetConfig):
    language: str = field(
        default='en',
        metadata=dict(help='Select which language subset in wikiann. '
                      'Refer to https://huggingface.co/datasets/wikiann .',
                      existence=True))


@DATASET.register_module('wikiann')
class Wikiann(BaseDataset):
    """Wikiann 为 NER 数据集，由 172 语言的子集组成，标签包括 LOC, ORG, PER。

    :param language: 选择哪个语言的子集
        参考 https://huggingface.co/datasets/wikiann
    """
    _config = WikiannConfig()

    def __init__(self, language: str = 'en', **kwargs):
        super().__init__(**kwargs)
        self.language = language

    def run(self):
        raw_dataset = load_dataset('wikiann', self.language)
        tag2idx = {
            'O': 0,
            'B-PER': 1,
            'I-PER': 2,
            'B-ORG': 3,
            'I-ORG': 4,
            'B-LOC': 5,
            'I-LOC': 6
        }
        idx2tag = {value: key for key, value in tag2idx.items()}
        datasets = {}

        for split, dataset in raw_dataset.items():
            split = split.replace('validation', 'dev')
            datasets[split] = DataSet()
            for sample in dataset:
                instance = Instance()
                instance.add_field('tokens', sample['tokens'])
                entity_mentions = []
                span = []
                current_tag = 0
                for i in np.arange(len(sample['ner_tags'])):
                    if sample['ner_tags'][i] != 0:
                        if len(span) == 0:
                            current_tag = sample['ner_tags'][i]
                            span.append(i)
                            continue
                        else:
                            if current_tag == sample['ner_tags'][i]:
                                span.append(i)
                                continue
                            else:
                                entity_mentions.append(
                                    (span, idx2tag[current_tag]))
                                span = [i]
                                current_tag = sample['ner_tags'][i]
                                continue
                    else:
                        if len(span) > 0:
                            entity_mentions.append(
                                (span, idx2tag[sample['ner_tags'][span[0]]]))
                            span = []
                if len(span) > 0:
                    entity_mentions.append(
                        (span, idx2tag[sample['ner_tags'][span[0]]]))
                instance.add_field('entity_mentions', entity_mentions)
                datasets[split].append(instance)

        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
