# -*- coding: UTF-8 -*- 
import json
from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from typing import Union, Sequence, Optional

from fastNLP import Evaluator, DataSet, Metric
from fastNLP.io import DataBundle

from fastie.controller.BaseController import BaseController, CONTROLLER
from fastie.envs import set_flag
from fastie.node import BaseNodeConfig


class InferenceMetric(Metric):

    def __init__(self, save_path: Optional[str] = None, verbose: bool = True):
        super().__init__(aggregate_when_get_metric=False)
        self.result: list = []
        self.save_path: Optional[str] = save_path
        self.verbose: bool = verbose

    def update(self, pred: Sequence[dict]):
        if self.save_path is not None:
            if self.backend.is_distributed() and self.backend.is_global_zero():
                with open(self.save_path, 'a+') as file:
                    file.write('\n'.join(
                        map(
                            lambda x: json.dumps(x),
                            reduce(lambda x, y: [*x, *y],
                                   self.all_gather_object(pred)))) + '\n')
            elif not self.backend.is_distributed():
                with open(self.save_path, 'a+') as file:
                    file.write('\n'.join(map(lambda x: json.dumps(x), pred)) +
                               '\n')
        if self.verbose and not self.backend.is_distributed():
            for sample in pred:
                # 判断一下不同的格式
                # 首先是 NER 小组约定的格式
                if 'entity_mentions' in sample.keys():
                    print('tokens: ', ' '.join(sample['tokens']))
                    print(
                        'pred:   ', ' '.join([
                            sample['tokens'][i] if i
                            in sample['entity_mentions'][0][0] else ''.join(
                                [' ' for j in range(len(sample['tokens'][i]))])
                            for i in range(len(sample['tokens']))
                        ]), f"  {sample['entity_mentions'][0][1]} -> "
                        f"{sample['entity_mentions'][0][2]}"
                        if len(sample['entity_mentions'][0]) == 3 else
                        f"  {sample['entity_mentions'][0][1]}")
                    if len(sample['entity_mentions']) > 1:
                        for entity_mention in sample['entity_mentions'][1:]:
                            print(
                                '        ', ' '.join([
                                    sample['tokens'][i]
                                    if i in entity_mention[0] else ''.join([
                                        ' ' for j in range(
                                            len(sample['tokens'][i]))
                                    ]) for i in range(len(sample['tokens']))
                                ]), f'  {entity_mention[1]} -> '
                                f'{entity_mention[2]}' if len(entity_mention)
                                == 3 else f'  {entity_mention[1]}')
                else:
                    # TODO: 其他类型的格式，例如为关系抽取小组制定的格式
                    pass
        self.result.extend(pred)

    def get_metric(self):
        return reduce(lambda x, y: x.extend(y),
                      self.all_gather_object(self.result))


def generate_step_fn(evaluator, batch):
    outputs = evaluator.evaluate_step(batch)
    content = '\n'.join(outputs['pred'])
    if getattr(evaluator, 'generate_save_path', None) is not None:
        with open(evaluator.generate_save_path, 'a+') as f:
            f.write(f'{content}\n')
    else:
        evaluator.result.extend(outputs['pred'])


@dataclass
class InferenceConfig(BaseNodeConfig):
    save_path: Optional[str] = field(
        default=None,
        metadata=dict(
            help='The path to save the generated results. If not set, output to '
            'the returned variable. ',
            existence=['infer']))
    verbose: bool = field(
        default=True,
        metadata=dict(
            help=
            'Whether to output the contents of each inference. Multiple cards '
            'are not supported. ',
            existence=['infer']))


@CONTROLLER.register_module('inference')
class Inference(BaseController):
    _config = InferenceConfig()

    def __init__(self,
                 save_path: Optional[str] = None,
                 verbose: bool = True,
                 **kwargs):
        super(Inference, self).__init__(**kwargs)
        self.save_path: Optional[str] = save_path
        self.verbose: bool = verbose
        set_flag('infer')

    def run(self,
            parameters_or_data: Optional[Union[dict, DataBundle, DataSet, str,
                                               Sequence[str]]] = None):
        parameters_or_data = BaseController.run(self, parameters_or_data)
        if parameters_or_data is None:
            print('Inference tool do not allow task and dataset to be left '
                  'empty. ')
            exit(1)
        parameters_or_data['evaluate_fn'] = 'inference_step'
        # parameters_or_data["evaluate_batch_step_fn"] = generate_step_fn
        parameters_or_data['verbose'] = False
        inference_metric = InferenceMetric(save_path=self.save_path,
                                           verbose=self.verbose)
        parameters_or_data['metrics'] = {'infer': inference_metric}
        evaluator = Evaluator(**parameters_or_data)
        # setattr(evaluator, "generate_save_path", self.save_path)
        # setattr(evaluator, "generate_result", [])
        evaluator.run()
        return inference_metric.result
