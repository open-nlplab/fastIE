"""Conll2003 like dataset for FastIE."""

__all__ = ['ColumnNER', 'ColumnNERConfig']

import os
from dataclasses import dataclass, field
from functools import reduce
from typing import Union, Sequence, List

from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle

from fastie.dataset.BaseDataset import DATASET, BaseDatasetConfig, BaseDataset


@dataclass
class ColumnNERConfig(BaseDatasetConfig):
    """ColumnNER 数据集配置类."""
    folder: str = field(
        default='',
        metadata=dict(help='The folder where the data set resides. \n'
                      'We will automatically read the possible train.txt, \n'
                      'dev.txt, test.txt and infer.txt in it. ',
                      existence=True))
    token_index: int = field(default=0,
                             metadata=dict(help='The index of tokens.',
                                           existence=True))
    tag_index: int = field(default=-1,
                           metadata=dict(help='The index of tags to predict.',
                                         existence=['train', 'eval']))
    split_char: str = field(
        default=' ',
        metadata=dict(help='The split char. If this parameter is not set, '
                      'it is separated by space. ',
                      existence=True))
    skip_content: str = field(
        default=' ',
        metadata=dict(help='The content to skip. If this item is not set, '
                      'it is divided by newline character. ',
                      existence=True))


@DATASET.register_module('column-ner')
class ColumnNER(BaseDataset):
    """Conll2003 like dataset for FastIE. Each row has a token and its
    corresponding NER tag.

    :param folder: The folder where the data set resides. ``train.txt``,
        ``dev.txt``, ``test.txt`` and ``infer.txt`` in the folder will be loaded.
    :param token_index: The index of tokens in a row.
    :param tag_index: The index of tags to predict in a row.
    :param split_char: The split character. If this parameter is not set, it is
        separated by space.
    :param skip_content: The content to skip. If this item is not set, it is
        divided by newline character.
    :param cache: Whether to cache the dataset.
    :param refresh_cache: Whether to refresh the cache.
    """
    _config = ColumnNERConfig()
    _help = 'Conll2003 like dataset for FastIE. Each row has a token and its corresponding NER tag.'

    def __init__(self,
                 folder: str = './',
                 token_index: int = 0,
                 tag_index: int = -1,
                 split_char: str = ' ',
                 skip_content: Union[str, Sequence[str]] = '\n',
                 cache: bool = False,
                 refresh_cache: bool = False,
                 **kwargs):
        super(ColumnNER, self).__init__(cache=cache,
                                        refresh_cache=refresh_cache,
                                        **kwargs)
        self.folder = folder
        self.token_index = token_index
        self.tag_index = tag_index
        self.split_char = split_char
        self.skip_content: Sequence = [skip_content] \
            if isinstance(skip_content, str) else skip_content

    def run(self) -> DataBundle:
        node = self

        class ColumnNERLoader(Loader):

            def _load(self, path: str) -> DataSet:
                ds = DataSet()
                data: List[dict] = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        if reduce(lambda x, y: x or y, [
                                line.startswith(content)
                                for content in node.skip_content
                        ]) or line.strip() == '':
                            if len(data) != 0:
                                if 'infer' in path:
                                    ds.append(
                                        Instance(
                                            tokens=[d['token'] for d in data]))
                                else:
                                    ...
                                # ds.append(
                                #     Instance(tokens=[d['token'] for d in data],
                                #              tags=[d['tag'] for d in data]))
                                # data = []
                            continue
                        lines = line.strip().split(node.split_char)
                        if 'infer' in path:
                            data.append({'token': lines[node.token_index]})
                        else:
                            data.append({
                                'token': lines[node.token_index],
                                'tag': lines[node.tag_index]
                            })
                if len(data) != 0:
                    if 'infer' in path:
                        ds.append(Instance(tokens=[d['token'] for d in data]))
                    else:
                        ...
                return ds

        data_bundle = ColumnNERLoader().load({
            file: os.path.exists(os.path.join(self.folder, f'{file}.txt'))
            for file in ('train', 'dev', 'test', 'infer')
            if os.path.exists(os.path.join(self.folder, f'{file}.txt'))
        })
        return data_bundle
