from fastie.controller import BaseController, CONTROLLER
from fastie.node import BaseNodeConfig
from fastie.envs import set_flag

from fastNLP import Evaluator, DataSet, Metric
from fastNLP.io import DataBundle

from typing import Union, Sequence, Optional
from dataclasses import dataclass, field
from functools import reduce

import json
import os

from dataclasses import dataclass



class InferenceMetric(Metric):
    def __init__(self, save_path: Optional[str]=None, verbose: bool=True):
        super().__init__(aggregate_when_get_metric=False)
        self.result = []
        self.save_path = save_path
        self.verbose = verbose

    def update(self, pred: Sequence[dict]):
        if self.save_path is not None:
            if self.backend.is_distributed() and self.backend.is_global_zero():
                with open(self.save_path, "a+") as file:
                    file.write("\n".join(map(lambda x: json.dumps(x), reduce(lambda x, y: x.extend(y), self.all_gather_object(pred)))) + "\n")
            elif not self.backend.is_distributed():
                with open(self.save_path, "a+") as file:
                    file.write("\n".join(map(lambda x: json.dumps(x), pred)) + "\n")
        if self.verbose and not self.backend.is_distributed():
            for sample in pred:
                for key, value in sample.items():
                    print(f"{key}: {' '.join(map(lambda x: f'{x}', list(value)))}\n")
        self.result.extend(pred)

    def get_metric(self):
        return reduce(lambda x, y: x.extend(y), self.all_gather_object(self.result))

def generate_step_fn(evaluator, batch):
    outputs = evaluator.evaluate_step(batch)
    content = '\n'.join(outputs['pred'])
    if getattr(evaluator, "generate_save_path", None) is not None:
        with open(evaluator.generate_save_path, "a+") as f:
            f.write(f"{content}\n")
    else:
        evaluator.result.extend(outputs['pred'])

@dataclass
class InferenceConfig(BaseNodeConfig):
    save_path: str = field(default=None,
                           metadata=dict(
                               help="The path to save the generated results. If not set, output to the returned variable. ",
                               existence=["infer"]
                           ))
    verbose: bool = field(default=True,
                            metadata=dict(
                                help="Whether to output the contents of each inference. Multiple cards are not supported. ",
                                existence=["infer"]
                            ))

@CONTROLLER.register_module("inference")
class Inference(BaseController):
    _config = InferenceConfig()
    def __init__(self,
                 save_path: Optional[str]=None,
                 verbose: bool=True,
                 **kwargs):
        BaseController.__init__(self, **kwargs)
        self.save_path = save_path
        self.verbose = verbose
        set_flag("infer")

    def run(self,
            parameters_or_data: Union[dict, DataBundle, DataSet, str, Sequence[str]] = dict()):
        parameters_or_data = BaseController.run(self, parameters_or_data)
        parameters_or_data["evaluate_fn"] = "inference_step"
        # parameters_or_data["evaluate_batch_step_fn"] = generate_step_fn
        parameters_or_data["verbose"] = False
        inference_metric = InferenceMetric(save_path=self.save_path, verbose=self.verbose)
        parameters_or_data["metrics"] = {
            "infer": inference_metric
        }
        evaluator = Evaluator(**parameters_or_data)
        # setattr(evaluator, "generate_save_path", self.save_path)
        # setattr(evaluator, "generate_result", [])
        evaluator.run()
        return inference_metric.result
