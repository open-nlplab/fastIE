import json
import os

from fastie.dataset.BaseDataset import BaseDataset, DATASET, BaseDatasetConfig

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.io import Loader, DataBundle

from dataclasses import dataclass, field
from typing import Union

@dataclass
class JsonNERConfig(BaseDatasetConfig):
    folder: str = field(default="",
                        metadata=dict(
                            help="The folder where the data set resides. "
                                 "We will automatically read the possible train.json, "
                                 "valid.json, test.json and infer.json in it. ",
                            existence=True
                        ))
    right_inclusive: bool = field(default=True,
                                  metadata=dict(
                                      help="When data is in the format of start and end, "
                                           "whether each span contains the token corresponding to end. ",
                                      existence=True
                                  ))

@DATASET.register_module("json-ner")
class JsonNER(BaseDataset):
    _config = JsonNERConfig()

    def __init__(self,
                 folder: str = "",
                 right_inclusive: bool = False,
                 cache: bool = False,
                 refresh_cache: bool = False,
                 tag_vocab: Union[Vocabulary, dict] = None,
                 **kwargs):
        BaseDataset.__init__(self, cache=cache, refresh_cache=refresh_cache, tag_vocab=tag_vocab, **kwargs)
        self.folder = folder
        self.right_inclusive = right_inclusive

    def run(self) -> DataBundle:
        vocabulary = Vocabulary()
        class JsonNERLoader(Loader):
            def _load(self, path: str) -> DataSet:
                dataset = DataSet()
                with open(path, "r", encoding="utf-8") as file:
                    for line in file.readlines():
                        line = line.strip()
                        if line:
                            sample: dict = json.loads(line)
                            instance = Instance()
                            instance.add_field("tokens", sample["tokens"])
                            if "entity_mentions" in sample.keys():
                                entity_mentions = []
                                for entity_mention in sample["entity_mentions"]:
                                    vocabulary.add_word(entity_mention["entity_type"])
                                    if "entity_index" in entity_mention.keys():
                                        entity_mentions.append((entity_mention["entity_index"], entity_mention["entity_type"]))
                                    elif "start" in entity_mention.keys() and "end" in entity_mention.keys():
                                        if self.right_inclusive:
                                            entity_mentions.append((list(range(entity_mention["start"], entity_mention["end"])), entity_mention["entity_type"]))
                                        else:
                                            entity_mentions.append((list(range(entity_mention["start"], entity_mention["end"] + 1)), entity_mention["entity_type"]))
                                instance.add_field("entity_mentions", entity_mentions)
                            dataset.append(instance)
                return dataset
        data_bundle = JsonNERLoader().load({file: os.path.exists(os.path.join(self.folder, f"{file}.json"))
                                           for file in ("train", "valid", "test", "infer")
                                           if os.path.exists(os.path.join(self.folder, f"{file}.json"))})
        self.tag_vocab = vocabulary._word2idx
        tagged_datasets: dict[str, DataSet] = {key: value for key, value in data_bundle.datasets.items() if key != "infer"}
        def index_tag(instance: Instance):
            entity_mentions = instance["entity_mentions"]
            return dict(entity_mentions=(entity_mentions[0], vocabulary.to_index(entity_mentions[1])))
        for dataset in tagged_datasets.values():
            dataset.apply_more(index_tag)
        return data_bundle
