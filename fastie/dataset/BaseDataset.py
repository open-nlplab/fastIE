import os
from dataclasses import dataclass, field
from types import MethodType
from typing import Union

from fastNLP import Vocabulary, cache_results

from fastie.envs import FASTIE_HOME
from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils import Registry

DATASET = Registry('DATASET')


def load_dataset(name, *args, **kwargs):
    return DATASET.get(name)(*args, **kwargs)


@dataclass
class BaseDatasetConfig(BaseNodeConfig):
    use_cache: bool = field(
        default=False,
        metadata=dict(
            help=
            'The result of data loading is cached for accelerated reading the next time it is used.',
            existence=True))
    refresh_cache: bool = field(
        default=False,
        metadata=dict(help='Clear cache (Use this when your data changes). ',
                      existence=True))


class BaseDataset(BaseNode):
    """数据集基类.

    Args:
        :use_cache (bool)[train,evaluation,inference]=False: 是否使用 cache.
        :refresh_cache (bool)[train,evaluation,inference]=False: 是否刷新 cache.
    """

    _config = BaseDatasetConfig()
    _help = '数据集基类'

    def __init__(self,
                 cache: bool = False,
                 refresh_cache: bool = False,
                 **kwargs):
        BaseNode.__init__(self, **kwargs)
        self.refresh_cache: bool = refresh_cache
        self.cache: bool = cache

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value: bool):
        if value:
            # 保存 cache 的位置默认为 `~/.fastie/cache/BaseDataset/cache.pkl`
            path = os.path.join(FASTIE_HOME,
                                f'cache/{self.__class__.__name__}/cache.pkl')
            object.__setattr__(
                self, 'run',
                cache_results(_cache_fp=f'{path}',
                              _refresh=self.refresh_cache)(self.run))
        self._cache = value

    def run(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run()
