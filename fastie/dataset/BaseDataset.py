"""
Base class for all FastIE datasets.
"""
__all__ = [
    'BaseDataset', 'BaseDatasetConfig', 'load_dataset', 'DATASET']
import os
import abc
from dataclasses import dataclass, field

from fastNLP import cache_results

from fastie.envs import FASTIE_HOME
from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils import Registry

DATASET = Registry('DATASET')


def load_dataset(name, *args, **kwargs):
    """
    根据 dataset 的注册名字加载 dataset 对象.
    :param name:
    :param args:
    :param kwargs:
    :return:
    """
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


class BaseDataset(BaseNode, metaclass=abc.ABCMeta):
    """
    FastIE 数据集基类.

    :param cache: 是否缓存数据集.
    :param refresh_cache: 是否刷新缓存.
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

    @abc.abstractmethod
    def run(self):
        """
        加载数据集, 返回一个 DataBundle 对象.
        :return:
        """
        raise NotImplementedError("The `run` method must be implemented. ")

    def __call__(self, *args, **kwargs):
        return self.run()
