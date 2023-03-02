from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset
from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.io import DataBundle

from fastie.dataset.BaseDataset import BaseDataset, DATASET, BaseDatasetConfig
from fastie.envs import logger


@dataclass
class Conll2003Config(BaseDatasetConfig):
    tag: str = field(default='ner',
                     metadata=dict(
                         help='Which fields in the dataset you want to use. '
                         'Select among: ner, pos, chunk.',
                         existence=['train', 'evaluate']))


@DATASET.register_module('conll2003')
class Conll2003(BaseDataset):
    """The shared task of CoNLL-2003 concerns language-independent named entity
    recognition."""
    _config = Conll2003Config()
    _help = 'Legacy dataset: conll2003. Refer to ' \
            'https://huggingface.co/datasets/conll2003 for more information.'

    def __init__(self,
                 tag: str = 'ner',
                 cache: bool = False,
                 refresh_cache: bool = False,
                 **kwargs):
        super(Conll2003, self).__init__(cache=cache,
                                        refresh_cache=refresh_cache,
                                        **kwargs)
        self.tag = tag

    def run(self):
        raw_dataset = load_dataset('conll2003')
        datasets = {}
        if self.tag not in ('ner'):
            logger.error('Tag must be `ner`.')
            exit(1)
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
        idx2tag = {'ner': {}}
        for i in np.arange(1, max(len(list(tag2idx['ner'].keys())), -1)):
            if i < len(list(tag2idx['ner'].keys())) and i % 2 == 0:
                idx2tag['ner'][i - 1] = list(
                    tag2idx['ner'].keys())[i].split('-')[1]
                idx2tag['ner'][i] = list(
                    tag2idx['ner'].keys())[i].split('-')[1]
        for split, dataset in raw_dataset.items():
            split = split.replace('validation', 'dev')
            datasets[split] = DataSet()
            for sample in dataset:
                instance = Instance()
                instance.add_field('tokens', sample['tokens'])
                if self.tag == 'ner':
                    entity_mentions = []
                    span = []
                    for i in np.arange(len(sample['ner_tags'])):
                        if sample['ner_tags'][i] != 0:
                            span.append(i)
                            continue
                        else:
                            if len(span) > 0:
                                entity_mentions.append((span, idx2tag['ner'][
                                    sample['ner_tags'][span[0]]]))
                                span = []
                    if len(span) > 0:
                        entity_mentions.append(
                            (span,
                             idx2tag['ner'][sample['ner_tags'][span[0]]]))
                    instance.add_field('entity_mentions', entity_mentions)
                datasets[split].append(instance)
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
