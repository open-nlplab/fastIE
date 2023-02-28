# -*- coding: UTF-8 -*- 
"""这个类还没写好，请勿参考."""

import os
from dataclasses import dataclass, field
from functools import reduce
from typing import Union, Sequence, Optional, List

from fastNLP import DataSet, Instance, Vocabulary

from fastNLP.io import Loader, DataBundle

from fastie.dataset.BaseDataset import DATASET, BaseDatasetConfig, BaseDataset
from fastie.envs import get_config


@dataclass
class ColumnNERConfig(BaseDatasetConfig):
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
                                         existence=['train', 'evaluate']))
    split_char: str = field(
        default=' ',
        metadata=dict(
            help=
            'The split char. If this parameter is not set, it is separated by '
            'space. ',
            existence=True))
    skip_content: str = field(
        default=' ',
        metadata=dict(
            help=
            'The content to skip. If this item is not set, it is divided by '
            'newline character. ',
            existence=True))


@DATASET.register_module('column-ner')
class ColumnNER(BaseDataset):
    """ 类似 conll2003 这种的数据集，每一行是一个 token 和对应的 tag
    Args:
        :folder (str)[train,evaluation,inference]=None: 数据集所在的文件夹.
        :token_index (int)[train,evaluation,inference]=0: token 所在的列.
        :tag_index (int)[train,evaluation,inference]=0: tag 所在的列.
        :split_char (str)[train,evaluation,inference]=" ": 分隔符.
        :skip_content (str)[train,evaluation,inference]=None: 跳过的行的内容.
    """
    _config = ColumnNERConfig()

    def __init__(self,
                 folder: str = './',
                 token_index: int = 0,
                 tag_index: int = -1,
                 split_char: str = ' ',
                 skip_content: Union[str, Sequence[str]] = '\n',
                 cache: bool = False,
                 refresh_cache: bool = False,
                 tag_vocab: Optional[Union[Vocabulary, dict]] = None,
                 **kwargs):
        super(ColumnNER, self).__init__(cache=cache,
                                        refresh_cache=refresh_cache,
                                        tag_vocab=tag_vocab,
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
                        ]):
                            if len(data) != 0:
                                ds.append(
                                    Instance(tokens=[d['token'] for d in data],
                                             tags=[d['tag'] for d in data]))
                                data = []
                            continue
                        lines = line.strip().split(node.split_char)
                        if 'generate' in path:
                            data.append({'token': lines[node.token_index]})
                        else:
                            data.append({
                                'token': lines[node.token_index],
                                'tag': lines[node.tag_index]
                            })
                if len(data) != 0:
                    ds.append(
                        Instance(tokens=[d['token'] for d in data],
                                 tags=[d['tag'] for d in data]))
                return ds

        data_bundle = ColumnNERLoader().load({
            file: os.path.exists(os.path.join(self.folder, f'{file}.txt'))
            for file in ('train', 'dev', 'test', 'generate')
            if os.path.exists(os.path.join(self.folder, f'{file}.txt'))
        })
        tagged_datasets = {
            key: value
            for key, value in data_bundle.datasets.items() if key != 'generate'
        }
        if len(tagged_datasets.keys()) > 0:
            tag_vocab = Vocabulary()
            if 'vocabulary' in get_config().keys():
                if get_config()['vocabulary'] != tag_vocab._word2idx:
                    print(
                        'Warning: Dictionaries generated from the dataset '
                        'are not the same as those in the provided preset,\n'
                        'and will be preferred to dictionaries generated from '
                        'the dataset')
            tag_vocab.from_dataset(tagged_datasets, field_name='tags')
            tag_vocab.index_dataset(tagged_datasets,
                                    field_name='tags',
                                    new_field_name='target')
            self.tag_vocab = tag_vocab._word2idx
        data_bundle.set_vocab(tag_vocab, 'tag')
        return data_bundle
