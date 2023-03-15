"""The shared dataset of CoNLL-2003 concerns language-independent named entity
recognition."""
__all__ = ['Conll2003', 'Conll2003Config']

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle

from fastie.dataset.BaseDataset import BaseDataset, DATASET, BaseDatasetConfig


@dataclass
class Conll2003Config(BaseDatasetConfig):
    """Conll2003 数据集的配置类."""
    pass


@DATASET.register_module('conll2003')
class Conll2003(BaseDataset):
    """The shared task of CoNLL-2003 concerns language-independent named entity
    recognition."""
    _config = Conll2003Config()
    _help = 'Conll2003 for NER task. Refer to ' \
            'https://huggingface.co/datasets/conll2003 for more information.'

    def __init__(self,
                 cache: bool = False,
                 refresh_cache: bool = False,
                 **kwargs):
        super(Conll2003, self).__init__(cache=cache,
                                        refresh_cache=refresh_cache,
                                        **kwargs)

    def run(self):
        raw_dataset = load_dataset('conll2003')
        datasets = {}
        tag2idx = {
            'ner': {
                'O': 0,
                'B-PER': 1,
                'I-PER': 2,
                'B-ORG': 3,
                'I-ORG': 4,
                'B-LOC': 5,
                'I-LOC': 6,
                'B-MISC': 7,
                'I-MISC': 8
            },
            'pos': {
                '"': 0,
                "''": 1,
                '#': 2,
                '$': 3,
                '(': 4,
                ')': 5,
                ',': 6,
                '.': 7,
                ':': 8,
                '``': 9,
                'CC': 10,
                'CD': 11,
                'DT': 12,
                'EX': 13,
                'FW': 14,
                'IN': 15,
                'JJ': 16,
                'JJR': 17,
                'JJS': 18,
                'LS': 19,
                'MD': 20,
                'NN': 21,
                'NNP': 22,
                'NNPS': 23,
                'NNS': 24,
                'NN|SYM': 25,
                'PDT': 26,
                'POS': 27,
                'PRP': 28,
                'PRP$': 29,
                'RB': 30,
                'RBR': 31,
                'RBS': 32,
                'RP': 33,
                'SYM': 34,
                'TO': 35,
                'UH': 36,
                'VB': 37,
                'VBD': 38,
                'VBG': 39,
                'VBN': 40,
                'VBP': 41,
                'VBZ': 42,
                'WDT': 43,
                'WP': 44,
                'WP$': 45,
                'WRB': 46
            },
            'chunk': {
                'O': 0,
                'B-ADJP': 1,
                'I-ADJP': 2,
                'B-ADVP': 3,
                'I-ADVP': 4,
                'B-CONJP': 5,
                'I-CONJP': 6,
                'B-INTJ': 7,
                'I-INTJ': 8,
                'B-LST': 9,
                'I-LST': 10,
                'B-NP': 11,
                'I-NP': 12,
                'B-PP': 13,
                'I-PP': 14,
                'B-PRT': 15,
                'I-PRT': 16,
                'B-SBAR': 17,
                'I-SBAR': 18,
                'B-UCP': 19,
                'I-UCP': 20,
                'B-VP': 21,
                'I-VP': 22
            }
        }
        idx2tag = {'ner': {}, 'pos': {}, 'chunk': {}}
        idx2tag['ner'] = {v: k for k, v in tag2idx['ner'].items()}
        idx2tag['pos'] = {v: k for k, v in tag2idx['pos'].items()}
        idx2tag['chunk'] = {v: k for k, v in tag2idx['chunk'].items()}
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
                            if current_tag == sample['ner_tags'][i] \
                                    or current_tag + 1 == sample['ner_tags'][i]:
                                span.append(i)
                                continue
                            else:
                                entity_mentions.append(
                                    (span, idx2tag['ner'][current_tag][2:]))
                                span = [i]
                                current_tag = sample['ner_tags'][i]
                                continue
                    else:
                        if len(span) > 0:
                            entity_mentions.append((span, idx2tag['ner'][
                                sample['ner_tags'][span[0]]][2:]))
                            span = []
                if len(span) > 0:
                    entity_mentions.append(
                        (span,
                         idx2tag['ner'][sample['ner_tags'][span[0]]][2:]))
                instance.add_field('entity_mentions', entity_mentions)
                instance.add_field('pos_tags', sample['pos_tags'])
                instance.add_field('chunk_tags', sample['chunk_tags'])
                instance.add_field('ner_tags', sample['ner_tags'])
                datasets[split].append(instance)
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
