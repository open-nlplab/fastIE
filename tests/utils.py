import os
import tempfile
from typing import Type

import pytest
from fastNLP.io import DataBundle

from fastie import Trainer, Evaluator, Inference, BaseTask


class UnifiedTaskTest:

    def setup_class(self,
                    task_cls: Type[BaseTask],
                    data_bundle: DataBundle,
                    extra_parameters: dict = {}):
        self.task_cls = task_cls
        self.extra_parameters = extra_parameters
        self.data_bundle = data_bundle

    @pytest.mark.parametrize('device', ['cpu', 'cuda:0', [0, 1]])
    def test_train(self, device):
        task = self.task_cls(**{
            'device': device,
            'batch_size': 2,
            'epoch': 2,
            **self.extra_parameters
        }).run(self.data_bundle)
        assert Trainer().run(task)

    @pytest.mark.parametrize('device', ['cpu', 'cuda:0', [0, 1]])
    def test_eval(self, device):
        task = self.task_cls(**{
            'device': device,
            'batch_size': 2,
            'epoch': 2,
            **self.extra_parameters
        }).run(self.data_bundle)
        assert Evaluator().run(task)

    @pytest.mark.parametrize('device', ['cpu', 'cuda:0', [0, 1]])
    def test_inference(self, device):
        task = self.task_cls(**{
            'device': device,
            'batch_size': 2,
            **self.extra_parameters
        }).run(self.data_bundle)
        assert Inference().run(task)

    def test_topk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self.task_cls(
                **{
                    'device': 'cpu',
                    'batch_size': 2,
                    'epoch': 2,
                    'topk': 2,
                    'save_model_folder': tmpdir,
                    **self.extra_parameters
                }).run(self.data_bundle)
            assert Trainer().run(task)
            assert len(os.listdir(tmpdir)) > 0

    def test_load_best_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self.task_cls(
                **{
                    'device': 'cpu',
                    'batch_size': 2,
                    'epoch': 2,
                    'load_best_model': True,
                    'save_model_folder': tmpdir,
                    **self.extra_parameters
                }).run(self.data_bundle)
            assert Trainer().run(task)
            assert len(os.listdir(tmpdir)) > 0

    # def test_fp16(self):
    #     task = self.task_cls(**{
    #         "device": "cpu",
    #         "batch_size": 2,
    #         "epoch": 2,
    #         "fp16": True,
    #         **self.extra_parameters
    #     }).run(self.data_bundle)
    #     assert Trainer().run(task)
