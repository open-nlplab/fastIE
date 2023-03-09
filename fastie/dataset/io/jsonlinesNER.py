"""JsonLinesNER dataset for FastIE."""
__all__ = ['JsonLinesNER', 'JsonLinesNERConfig']
import json
import os

from fastie.dataset.BaseDataset import BaseDataset, DATASET, BaseDatasetConfig

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.io import Loader, DataBundle

from dataclasses import dataclass, field
from typing import Union, Dict


@dataclass
class JsonLinesNERConfig(BaseDatasetConfig):
    """JsonLinesNER 数据集配置类."""
    folder: str = field(
        default='',
        metadata=dict(help='The folder where the data set resides. '
                      'We will automatically read the possible train.jsonl, '
                      'dev.jsonl, test.jsonl and infer.jsonl in it. ',
                      existence=True))
    right_inclusive: bool = field(
        default=True,
        metadata=dict(
            help='When data is in the format of start and end, '
            'whether each span contains the token corresponding to end. ',
            existence=True))


@DATASET.register_module('jsonlines-ner')
class JsonLinesNER(BaseDataset):
    """JsonLinesNER dataset for FastIE. Each row has a NER sample in json
    format:

    .. code-block:: json
    {
        "tokens": ["I", "love", "FastIE", "."],
        "entity_mentions": [
            {
                "entity_index": [2],
                "entity_type": "MISC"
            },
    }

    or:

    .. code-block:: json
    {
        "tokens": ["I", "love", "FastIE", "."],
        "entity_mentions": [
            {
                "start": 2,
                "end": 3,
                "entity_type": "MISC"
            },
    }

    :param folder: The folder where the data set resides.
    :param right_inclusive: When data is in the format of start and end,
        whether each span contains the token corresponding to end.
    :param cache: Whether to cache the dataset.
    :param refresh_cache: Whether to refresh the cache.
    """
    _config = JsonLinesNERConfig()
    _help = 'JsonLinesNER dataset for FastIE. Each row has a NER sample in json format. '

    def __init__(self,
                 folder: str = '',
                 right_inclusive: bool = False,
                 cache: bool = False,
                 refresh_cache: bool = False,
                 **kwargs):
        BaseDataset.__init__(self,
                             cache=cache,
                             refresh_cache=refresh_cache,
                             **kwargs)
        self.folder = folder
        self.right_inclusive = right_inclusive

    def run(self) -> DataBundle:
        vocabulary = Vocabulary()
        node = self

        class JsonNERLoader(Loader):

            def _load(self, path: str) -> DataSet:
                dataset = DataSet()
                with open(path, 'r', encoding='utf-8') as file:
                    for line in file.readlines():
                        line = line.strip()
                        if line:
                            sample: dict = json.loads(line)
                            instance = Instance()
                            instance.add_field('tokens', sample['tokens'])
                            if 'entity_mentions' in sample.keys():
                                entity_mentions = []
                                for entity_mention in sample[
                                        'entity_mentions']:
                                    vocabulary.add_word(
                                        entity_mention['entity_type'])
                                    if 'entity_index' in entity_mention.keys():
                                        entity_mentions.append(
                                            (entity_mention['entity_index'],
                                             entity_mention['entity_type']))
                                    elif 'start' in entity_mention.keys(
                                    ) and 'end' in entity_mention.keys():
                                        if node.right_inclusive:
                                            entity_mentions.append((list(
                                                range(
                                                    entity_mention['start'],
                                                    entity_mention['end'] + 1)
                                            ), entity_mention['entity_type']))
                                        else:
                                            entity_mentions.append((list(
                                                range(entity_mention['start'],
                                                      entity_mention['end'])
                                            ), entity_mention['entity_type']))
                                instance.add_field('entity_mentions',
                                                   entity_mentions)
                            dataset.append(instance)
                return dataset

        data_bundle = JsonNERLoader().load({
            file: os.path.join(self.folder, f'{file}.jsonl')
            for file in ('train', 'dev', 'test', 'infer')
            if os.path.exists(os.path.join(self.folder, f'{file}.jsonl'))
        })
        return data_bundle
