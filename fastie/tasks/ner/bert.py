from dataclasses import dataclass, field
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from fastNLP import prepare_dataloader, Instance, Vocabulary
from fastNLP.core.metrics import Accuracy
from fastNLP.io import DataBundle
from fastNLP.transformers.torch.models.bert import BertModel, BertConfig, BertTokenizer
from torch import nn

from fastie.envs import get_flag
from fastie.tasks import BaseTask, BaseTaskConfig, NER

from typing import Union, Sequence, Callable, Generator, Optional


class Model(nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path: str = None,
                 num_labels: int = None,
                 tag_vocab: Vocabulary = None,
                 **kwargs):
        super(Model, self).__init__()
        if pretrained_model_name_or_path is not None:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        else:
            self.bert = BertModel(BertConfig(**kwargs))
        self.num_labels = num_labels
        self.classificationHead = nn.Linear(self._get_bert_embedding_dim(), num_labels)
        self.tag_vocab = tag_vocab

    def _get_bert_embedding_dim(self):
        with torch.no_grad():
            temp = torch.zeros(1, 1).int().to(self.bert.device)
            return self.bert(temp).last_hidden_state.shape[-1]

    def forward(self, input_ids, attention_mask):
        features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        features = self.classificationHead(features)
        return dict(features=features)

    def train_step(self, input_ids, attention_mask, offset_mask, entity_mentions):
        features = self.forward(input_ids, attention_mask)["features"]
        loss = 0
        for b in range(features.shape[0]):
            logits = F.softmax(features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            for entity_mention in entity_mentions[b]:
                target = torch.zeros(self.num_labels).to(features.device)
                target[entity_mention[1]] = 1
                for i in entity_mention[0]:
                    loss += F.binary_cross_entropy(logits[i], target)
        return dict(loss=loss)

    def evaluate_step(self, input_ids, attention_mask, offset_mask, entity_mentions):
        features = self.forward(input_ids, attention_mask)["features"]
        pred_list = []
        target_list = []
        max_len = 0
        for b in range(features.shape[0]):
            logits = F.softmax(features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            pred = logits.argmax(dim=1).to(features.device)
            target = torch.zeros(pred.shape[0]).to(features.device)
            if pred.shape[0] > max_len:\
                max_len = pred.shape[0]
            for entity_mention in entity_mentions[b]:
                for i in entity_mention[0]:
                    target[i] = entity_mention[1]
            pred_list.append(pred)
            target_list.append(target)
        pred = torch.stack([F.pad(pred, (0, max_len - pred.shape[0])) for pred in pred_list])
        target = torch.stack([F.pad(target, (0, max_len - target.shape[0])) for target in target_list])
        return dict(pred=pred, target=target)

    def inference_step(self, tokens, input_ids, attention_mask, offset_mask):
        features = self.forward(input_ids, attention_mask)["features"]
        pred_list = []
        for b in range(features.shape[0]):
            logits = F.softmax(features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            pred = logits.argmax(dim=1).to(features.device)
            pred_list.append({
                "tokens": tokens[b],
                "label": [self.tag_vocab.idx2word[int(i)] for i in pred],
                "possibility": [round(float(i.max()), 3) for i in logits]
            })
        # 推理的结果一定是可 json 化的，建议 List[Dict]，比如这里输出了预测的标签和置信度
        return dict(pred=pred_list)


@dataclass
class BertNERConfig(BaseTaskConfig):
    pretrained_model_name_or_path: str = field(default="bert-base-uncased",
                                               metadata=dict(
                                                   help="name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for options).",
                                                   existence=True
                                               ))
    num_labels: int = field(default=9,
                            metadata=dict(
                                help="Number of label categories to predict.",
                                existence=True
                            ))


@NER.register_module("bert")
class BertNER(BaseTask):
    """ 用预训练模型 Bert 对 toke 进行向量表征，然后通过 classification head 对每个 token 进行分类。
    """
    # 必须在这里定义自己 config
    _config = BertNERConfig()
    # 帮助信息，会显示在命令行分组的帮助信息中
    _help = "Use pre-trained BERT and a classification head to classify tokens"
    def __init__(self,
                 pretrained_model_name_or_path: str = "bert-base-uncased",
                 num_labels: int = 9,
                 tag_vocab: Union[dict, Vocabulary] = None,
                 # 以下是父类的参数，也要复制过来，可以查看一下 BaseTask 参数
                 cuda: Union[bool, int, Sequence[int]] = False,
                 batch_size: int = 32,
                 lr: float = 2e-5,
                 load_model: str = None,
                 **kwargs):
        # 必须要把父类 （BaseTask）的参数也复制过来，否则用户没有父类的代码提示；
        # 在这里进行父类的初始化；
        # 这里的 cuda 不需要我们管，父类会设置好 parameters 里面的 device。
        BaseTask.__init__(self, cuda=cuda, batch_size=batch_size, lr=lr, load_model=load_model, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.num_labels = num_labels
        self.tag_vocab = tag_vocab

    def run(self, data_bundle: DataBundle):
        # 注意，接收的 data_bundle 可能有用来 infer 的，也就是说没有 label 信息，预处理的时候要注意
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        _tag_vocab = Vocabulary(padding=None, unknown=None)
        # 允许用户传入自己的 tag2idx 字典，允许多种格式
        if self.tag_vocab is not None:
            if isinstance(self.tag_vocab, dict) and isinstance(list(self.tag_vocab.keys())[0], str):
                _tag_vocab._word2idx = self.tag_vocab
                _tag_vocab._idx2word = {v: k for k, v in self.tag_vocab.items()}
            elif isinstance(self.tag_vocab, dict) and isinstance(list(self.tag_vocab.keys())[0], int):
                _tag_vocab._idx2word = self.tag_vocab
                _tag_vocab._word2idx = {v: k for k, v in self.tag_vocab.items()}
            elif isinstance(self.tag_vocab, list):
                _tag_vocab.from_dataset(*data_bundle.datasets.values(), field_name="target")
            else:
                self.tag_vocab = None
        # 如果用户不传入 tag2idx 字典，那么我们自己构建
        if self.tag_vocab is None:
            def construct_vocab(instance: Instance):
                # 当然，用来 infer 的数据集是无法构建的，这里判断以下
                if "entity_mentions" in instance.keys():
                    for entity_mention in instance["entity_mentions"]:
                        _tag_vocab.add(entity_mention[1])
                return instance
            data_bundle.apply_more(construct_vocab)
        # 将 token 转换为 id，由于 bpe 会改变 token 的长度，所以除了 input_ids，
        # attention_mask 意外还返回了 offset_mask，input_ids 里面哪些是原来的 token
        def tokenize(instance: Instance):
            result_dict = {}
            input_ids_list, attention_mask_list, offset_mask_list = [], [], []
            for token in instance["tokens"]:
                tokenized_token = self.tokenizer([token],
                                                 is_split_into_words=True,
                                                 return_tensors="np",
                                                 return_attention_mask=True,
                                                 return_token_type_ids=False,
                                                 add_special_tokens=False)
                token_offset_mask = np.zeros(tokenized_token["input_ids"].shape, dtype=int)
                token_offset_mask[0, 0] = 1
                input_ids_list.append(tokenized_token["input_ids"])
                attention_mask_list.append(tokenized_token["attention_mask"])
                offset_mask_list.append(token_offset_mask)
            input_ids = reduce(lambda x, y: np.concatenate((x, y), axis=1), input_ids_list)[0, :]
            attention_mask = reduce(lambda x, y: np.concatenate((x, y), axis=1), attention_mask_list)[0, :]
            offset_mask = reduce(lambda x, y: np.concatenate((x, y), axis=1), offset_mask_list)[0, :]
            result_dict["input_ids"] = input_ids
            result_dict["attention_mask"] = attention_mask
            result_dict["offset_mask"] = offset_mask
            # 顺便把 tag 转化为 id
            if "entity_mentions" in instance.keys():
                for i in range(len(instance["entity_mentions"])):
                    instance["entity_mentions"][i] = (instance["entity_mentions"][i][0], _tag_vocab.to_index(instance["entity_mentions"][i][1]))
                result_dict["entity_mentions"] = instance["entity_mentions"]
            return result_dict

        data_bundle.apply_more(tokenize)
        # 为了 infer 的时候能够使用，我们把 tag_vocab 存起来
        model = Model(self.pretrained_model_name_or_path, self.num_labels, _tag_vocab)
        # Vocabulary 这种复杂的数据类型是不会存到配置文件中的，所以把 dict 类型的 word2idx 存起来
        self.tag_vocab = _tag_vocab._word2idx
        metrics = {
            "accuracy": Accuracy()
        }
        # 使用 get_flag 判断现在要进行的事情，可能的取值有 `train`, `eval`, `infer`, `interact`
        if get_flag() == "train":
            train_dataloader = prepare_dataloader(data_bundle.get_dataset("train"),
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
            # 用户不一定需要验证集，所以这里要判断一下
            if "valid" in data_bundle.datasets.keys():
                evaluate_dataloader = prepare_dataloader(data_bundle.get_dataset("valid"),
                                                         batch_size=self.batch_size,
                                                         shuffle=True)
                parameters = dict(model=model,
                                  optimizers=torch.optim.Adam(model.parameters(), lr=self.lr),
                                  train_dataloader=train_dataloader,
                                  evaluate_dataloaders=evaluate_dataloader,
                                  metrics=metrics)
            else:
                parameters = dict(model=model,
                                  optimizers=torch.optim.Adam(model.parameters(), lr=self.lr),
                                  train_dataloader=train_dataloader,
                                  metrics=metrics)
        elif get_flag() == "eval":
            evaluate_dataloader = prepare_dataloader(data_bundle.get_dataset("valid"),
                                                     batch_size=self.batch_size,
                                                     shuffle=True)
            parameters = dict(
                model=model,
                evaluate_dataloaders=evaluate_dataloader,
                metrics=metrics
            )
        elif get_flag() == "infer":
            # 注意：infer 和 eval 其实并没有区别，只是把 evaluate_dataloaders 换成推理的数据集了；
            # 我们不需要管怎么推理的，只要在模型里面写好 inference_step 就可以了
            infer_dataloader = prepare_dataloader(data_bundle.get_dataset("infer"),
                                                     batch_size=self.batch_size,
                                                     shuffle=True)
            parameters = dict(
                model=model,
                evaluate_dataloaders=infer_dataloader,
                driver='torch'
            )
        return parameters
