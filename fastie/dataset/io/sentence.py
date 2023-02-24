from fastie.dataset import DATASET, BaseDataset, BaseDatasetConfig

from typing import Union, Sequence, Optional

from dataclasses import dataclass, field

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.io import DataBundle


@dataclass
class SentenceConfig(BaseDatasetConfig):
    sentence: str = field(default='',
                          metadata=dict(help='Input a sequence as a dataset.',
                                        existence=True,
                                        nargs='+',
                                        multi_method='space-join'))


@DATASET.register_module('sentence')
class Sentence(BaseDataset):
    _config = SentenceConfig()
    _help = 'Input a sequence as a dataset. (Only for inference). '

    def __init__(self,
                 sentence: Optional[Union[Sequence[str], str]] = None,
                 cache: bool = False,
                 refresh_cache: bool = False,
                 tag_vocab: Optional[Union[Vocabulary, dict]] = None,
                 **kwargs):
        BaseDataset.__init__(self,
                             cache=cache,
                             refresh_cache=refresh_cache,
                             tag_vocab=tag_vocab,
                             **kwargs)
        self.sentence = sentence

    def run(self):
        dataset = DataSet()
        sentences = [self.sentence] if isinstance(self.sentence,
                                                  str) else self.sentence
        for sentence in sentences:
            dataset.append(Instance(tokens=sentence.split(' ')))
        data_bundle = DataBundle(datasets={'infer': dataset})
        return data_bundle
