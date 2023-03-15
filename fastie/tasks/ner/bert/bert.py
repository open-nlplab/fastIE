"""BertNER."""
__all__ = ['BertNER', 'BertNERConfig']

from dataclasses import dataclass, field
from functools import reduce
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from fastNLP import Instance, Vocabulary
from fastNLP.core.metrics import Accuracy
from fastNLP.io import DataBundle
from fastNLP.transformers.torch.models.bert import BertModel, BertConfig, \
    BertTokenizer
from torch import nn

from fastie.tasks.BaseTask import NER
from fastie.tasks.ner.BaseNERTask import BaseNERTask, BaseNERTaskConfig


class Model(nn.Module):

    def __init__(self,
                 pretrained_model_name_or_path: Optional[str] = None,
                 num_labels: int = 9,
                 tag_vocab: Optional[Vocabulary] = None,
                 **kwargs):
        super(Model, self).__init__()
        if pretrained_model_name_or_path is not None:
            self.bert = BertModel.from_pretrained(
                pretrained_model_name_or_path)
        else:
            self.bert = BertModel(BertConfig(**kwargs))
        self.bert.requires_grad_(False)
        self.num_labels = num_labels
        self.classificationHead = nn.Linear(self._get_bert_embedding_dim(),
                                            num_labels)
        # 为了推理过程中能输出人类可读的结果，把 tag2label 也传进来
        self.tag_vocab = tag_vocab

    def _get_bert_embedding_dim(self):
        with torch.no_grad():
            temp = torch.zeros(1, 1).int().to(self.bert.device)
            return self.bert(temp).last_hidden_state.shape[-1]

    def forward(self, input_ids, attention_mask):
        features = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
        features = self.classificationHead(features)
        return dict(features=features)

    def train_step(self, input_ids, attention_mask, offset_mask,
                   entity_mentions):
        features = self.forward(input_ids, attention_mask)['features']
        loss = 0
        for b in range(features.shape[0]):
            logits = F.softmax(
                features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            target = torch.zeros(logits.shape[0]).to(features.device)
            for entity_mention in entity_mentions[b]:
                for i in entity_mention[0]:
                    target[i] = entity_mention[1]
            for i in range(logits.shape[0]):
                one_hot_target = torch.zeros(self.num_labels).to(
                    features.device)
                one_hot_target[int(target[i])] = 1
                loss += F.binary_cross_entropy(logits[i], one_hot_target)
        return dict(loss=loss)

    def evaluate_step(self, input_ids, attention_mask, offset_mask,
                      entity_mentions):
        features = self.forward(input_ids, attention_mask)['features']
        pred_list = []
        target_list = []
        max_len = 0
        for b in range(features.shape[0]):
            logits = F.softmax(
                features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            pred = logits.argmax(dim=1).to(features.device)
            target = torch.zeros(pred.shape[0]).to(features.device)
            if pred.shape[0] > max_len:
                max_len = pred.shape[0]
            for entity_mention in entity_mentions[b]:
                for i in entity_mention[0]:
                    target[i] = entity_mention[1]
            pred_list.append(pred)
            target_list.append(target)
        pred = torch.stack(
            [F.pad(pred, (0, max_len - pred.shape[0])) for pred in pred_list])
        target = torch.stack([
            F.pad(target, (0, max_len - target.shape[0]))
            for target in target_list
        ])
        return dict(pred=pred, target=target)

    def inference_step(self, tokens, input_ids, attention_mask, offset_mask):
        features = self.forward(input_ids, attention_mask)['features']
        pred_list = []
        for b in range(features.shape[0]):
            logits = F.softmax(
                features[b][offset_mask[b].nonzero(), :].squeeze(1), dim=1)
            pred = logits.argmax(dim=1).to(features.device)
            pred_dict = {}
            pred_dict['tokens'] = tokens[b]
            pred_dict['entity_mentions'] = []
            for i in range(pred.shape[0]):
                # 考虑一下，如果用户没有传入 tag_vocab，那么这里的输出就是 idx
                if self.tag_vocab is not None:
                    pred_dict['entity_mentions'].append(
                        ([i], self.tag_vocab.idx2word[int(pred[i])],
                         round(float(logits[i].max()), 3)))
                else:
                    pred_dict['entity_mentions'].append(
                        ([i], int(pred[i]), round(float(logits[i].max()), 3)))
            pred_list.append(pred_dict)
        # 推理的结果一定是可 json 化的，建议 List[Dict]，和输入的数据集的格式一致
        # 这里的结果是用户可读的，所以建议把 idx2label 存起来
        # 怎么存可以看一下下面 233 行
        return dict(pred=pred_list)


@dataclass
class BertNERConfig(BaseNERTaskConfig):
    """BertNER 所需参数."""
    pretrained_model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata=dict(
            help='name of transformer model (see '
            'https://huggingface.co/transformers/pretrained_models.html for '
            'options).',
            existence='train'))
    lr: float = field(default=2e-5,
                      metadata=dict(help='learning rate', existence='train'))


@NER.register_module('bert')
class BertNER(BaseNERTask):
    """BertNER 使用预训练的 BERT 模型和分类头来做 NER 任务.

    :param pretrained_model_name_or_path: transformers 预训练 BERT 模型名字或路径.
        (see https://huggingface.co/models for options).
    :param lr: 学习率
    """
    # 必须在这里定义自己 config
    _config = BertNERConfig()
    # 帮助信息，会显示在命令行分组的帮助信息中
    _help = 'Use pre-trained BERT and a classification head to classify tokens'

    def __init__(self,
                 pretrained_model_name_or_path: str = 'bert-base-uncased',
                 lr: float = 2e-5,
                 **kwargs):
        # 必须要把父类 （BaseTask）的参数也复制过来，否则用户没有父类的代码提示；
        # 在这里进行父类的初始化；
        # 父类的参数我们不需要进行任何操作，比如这里的 cuda 和 load_model，我们无视就可以了。
        super().__init__(**kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lr = lr

    def on_dataset_preprocess(self, data_bundle: DataBundle,
                              tag_vocab: Dict[str, Vocabulary],
                              state_dict: Optional[dict]) -> DataBundle:
        """数据预处理, 包括将 `label` 通过生成或加载的 `tag_vocab` 转化为 id, 并将 `tokens` 通过
        `BertTokenizer` 转化为 id.

        :param data_bundle: 原始数据
        :param tag_vocab: 生成或加载的 `tag_vocab`
        :param state_dict: 加载的 `checkpoint`
        :return: 处理后的数据集
        """
        # 数据预处理
        tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_model_name_or_path)

        def tokenize(instance: Instance):
            result_dict = {}
            input_ids_list, attention_mask_list, offset_mask_list = [], [], []
            for token in instance['tokens']:
                tokenized_token = tokenizer([token],
                                            is_split_into_words=True,
                                            return_tensors='np',
                                            return_attention_mask=True,
                                            return_token_type_ids=False,
                                            add_special_tokens=False)
                token_offset_mask = np.zeros(
                    tokenized_token['input_ids'].shape, dtype=int)
                token_offset_mask[0, 0] = 1
                input_ids_list.append(tokenized_token['input_ids'])
                attention_mask_list.append(tokenized_token['attention_mask'])
                offset_mask_list.append(token_offset_mask)
            input_ids = reduce(lambda x, y: np.concatenate((x, y), axis=1),
                               input_ids_list)[0, :]
            attention_mask = reduce(
                lambda x, y: np.concatenate((x, y), axis=1),
                attention_mask_list)[0, :]
            offset_mask = reduce(lambda x, y: np.concatenate((x, y), axis=1),
                                 offset_mask_list)[0, :]
            result_dict['input_ids'] = input_ids
            result_dict['attention_mask'] = attention_mask
            result_dict['offset_mask'] = offset_mask
            # 顺便把 tag 转化为 id
            if 'entity_mentions' in instance.keys():
                for i in range(len(instance['entity_mentions'])):
                    instance['entity_mentions'][i] = (
                        instance['entity_mentions'][i][0],
                        tag_vocab['entity'].to_index(
                            instance['entity_mentions'][i][1]))
                result_dict['entity_mentions'] = instance['entity_mentions']
            return result_dict

        data_bundle.apply_more(tokenize)
        return data_bundle

    def on_setup_model(self, data_bundle: DataBundle,
                       tag_vocab: Dict[str, Vocabulary],
                       state_dict: Optional[dict]):
        """加载 BERT 模型和分类头.

        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 生成或加载的 `tag_vocab`
        :param state_dict: 加载的 `checkpoint`
        :return: 拥有 ``train_step``、``evaluate_step``、 ``inference_step``
            方法的对象
        """
        # 模型加载阶段
        model = Model(self.pretrained_model_name_or_path,
                      num_labels=len(list(
                          tag_vocab['entity'].word2idx.keys())),
                      tag_vocab=tag_vocab['entity'])
        if state_dict:
            model.load_state_dict(state_dict['model'])
        return model

    def on_setup_optimizers(self, model, data_bundle: DataBundle,
                            tag_vocab: Dict[str, Vocabulary],
                            state_dict: Optional[dict]):
        """加载 `Adam` 优化器.

        :param model: 模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 生成或加载的 `tag_vocab`
        :param state_dict: 加载的 `checkpoint`
        :return:
        """
        # 优化器加载阶段
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def on_setup_metrics(self, model, data_bundle: DataBundle,
                         tag_vocab: Dict[str, Vocabulary],
                         state_dict: Optional[dict]) -> dict:
        """加载 `Accuracy` 评价指标.

        :param model: 模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 生成或加载的 `tag_vocab`
        :param state_dict: 加载的 `checkpoint`
        :return:
        """
        # 评价指标加载阶段
        return {'accuracy': Accuracy()}
