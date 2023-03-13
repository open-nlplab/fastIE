"""Base class for all FastIE tasks."""

__all__ = ['BaseTask', 'BaseTaskConfig', 'NER', 'RE', 'EE']

import abc
import os.path
from dataclasses import dataclass, field
from typing import Sequence, Union, Generator, Optional, Any

from fastNLP import Vocabulary, Callback, prepare_dataloader
from fastNLP.core.callbacks import CheckpointCallback, LoadBestModelCallback
from fastNLP.io import DataBundle

from fastie.envs import get_flag, logger
from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils.hub import Hub
from fastie.utils.registry import Registry
from fastie.utils.utils import inspect_metrics

NER = Registry('NER')
RE = Registry('RE')
EE = Registry('EE')


@dataclass
class BaseTaskConfig(BaseNodeConfig, metaclass=abc.ABCMeta):
    load_model: str = field(
        default='',
        metadata=dict(help='Load model from file path or model name. ',
                      existence=True))
    save_model_folder: str = field(
        default='',
        metadata=dict(
            help=
            'The folder to save the model when using `topk` or `load_best_model`. '
            'If not set, all models will be saved in the current directory.',
            existence='train'))
    batch_size: int = field(default=32,
                            metadata=dict(help='Batch size. ', existence=True))
    epochs: int = field(default=20,
                        metadata=dict(help='Total number of training epochs. ',
                                      existence='train'))
    evaluate_every: int = field(
        default=-1,
        metadata=dict(help='Frequency of evaluation. '
                      'If the value is positive, '
                      'the evaluation is performed once every n batch. '
                      'If the value is negative, '
                      'the evaluation is performed once every n epochs ',
                      existence='train'))
    monitor: str = field(
        default='',
        metadata=dict(help='The metric name which is used to select '
                      'the best model when using `topk` or `load_best_model`. '
                      'If not set, the first metric will be used',
                      existence='train'))
    is_large_better: str = field(default='True',
                                 metadata=dict(
                                     help='Is the `metric` monitored by '
                                     '`monitor` the larger the better. ',
                                     existence='train'))
    topk: int = field(
        default=0,
        metadata=dict(
            help='Save the top-k model to `save_model_folder`/fastie_topk_model '
            'according to the `monitor`. ',
            existence='train'))
    load_best_model: bool = field(
        default=False,
        metadata=dict(
            help=
            'Save the best model to `save_model_folder`/fastie_best_model according to the `monitor`. ',
            existence='train'))

    fp16: bool = field(default=False,
                       metadata=dict(help='Enable mixed-precision training. ',
                                     existence='train'))

    device: str = field(
        default='cpu',
        metadata=dict(
            help=
            'Specifies the devices to use. If you want to use multiple GPUs, connect the GPU id with a `,`',
            existence=True))


class BaseTask(BaseNode, metaclass=abc.ABCMeta):
    """FastIE 的任务基类，所有任务需要继承基类并重写生命周期方法。

    :param load_model: 模型文件的路径或者模型名
    :param save_model_folder: ``topk`` 或 ``load_best_model`` 保存模型的文件夹
    :param batch_size: batch size
    :param epochs: 训练的轮数
    :param monitor: 根据哪个 ``metric`` 选择  ``topk`` 和 ``load_best_model``；
        如果不设置，则默认使用结果中的第一个 ``metric``
    :param is_large_better: ``metric`` 中 ``monitor`` 监控的指标是否越大越好
    :param topk: 将 ``metric`` 中 ``monitor`` 监控的指标最好的 k 个模型保存到
        ``save_model_folder`` 中
    :param load_best_model: 是否在训练结束后将 ``metric`` 中 ``monitor`` 监控的指标最
        好的模型保存到 ``save_model_folder`` 中，并自动加载到 ``task`` 中
    :param fp16: 是否使用混合精度训练
    :param evaluate_every: 训练过程中检验的频率,
        ``topk`` 和 ``load_best_model`` 将在所有的检验中选择:
            * 为 ``0`` 时则训练过程中不进行检验
            * 如果为正数，则每 ``evaluate_every`` 个 batch 进行一次检验
            * 如果为负数，则每 ``evaluate_every`` 个 epoch 进行一次检验
    :param device: 指定具体训练时使用的设备
        device 的可选输入如下所示:
            * *str*: 例如 ``'cpu'``, ``'cuda'``, ``'cuda:0'``, ``'cuda:1'``,
            `'gpu:0'`` 等；
            * *int*: 将使用 ``device_id`` 为该值的 ``gpu`` 进行训练；如果值为 -1，那么
            默认使用全部的显卡；
            * *list(int)*: 如果多于 1 个device，应当通过该种方式进行设定，将会启用分布式程序。
    """
    _config = BaseTaskConfig()
    _help = 'Base class for all tasks. '

    def __init__(self,
                 load_model: str = '',
                 save_model_folder: str = '',
                 batch_size: int = 32,
                 epochs: int = 20,
                 monitor: str = '',
                 is_large_better: bool = True,
                 topk: int = 0,
                 load_best_model: bool = False,
                 fp16: bool = False,
                 evaluate_every: int = -1,
                 device: Union[int, Sequence[int], str] = 'cpu',
                 **kwargs):
        BaseNode.__init__(self, **kwargs)
        self.load_model = load_model
        self.save_model_folder = save_model_folder
        self.epochs = epochs
        self.topk = topk
        self.load_best_model = load_best_model
        self.monitor = monitor
        self.is_large_better = is_large_better
        self.fp16 = fp16
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.device = device

        object.__setattr__(self, 'run', self._run_generator())

        self._on_generate_and_check_tag_vocab_cache: Any = None
        self._on_dataset_preprocess_cache: Any = None
        self._on_setup_model_cache: Any = None
        self._on_setup_optimizers_cache: Any = None
        self._on_setup_dataloader_cache: dict = {}
        self._on_setup_callbacks_cache: Union[Callback, Sequence[Callback]] \
            = []
        self._on_setup_metrics_cache: Any = None
        self._on_setup_extra_fastnlp_parameters_cache: Any = None
        self._on_get_state_dict_cache: Any = None

    @abc.abstractmethod
    def on_generate_and_check_tag_vocab(self,
                                        data_bundle: DataBundle,
                                        state_dict: Optional[dict]) \
            -> Optional[Vocabulary]:
        """根据数据集中的已标注数据的标签词频生成标签词典。 如果加载模型得到的 ``state_dict`` 中存在
        ``tag_vocab``，则检查是否与根据 ``data_bundle`` 生成的 tag_vocab 一致 (优先使用加载得到的
        tag_vocab)。

        :param data_bundle: 原始数据集，
            可能包含 ``train``、``dev``、``test``、``infer`` 四种，需要分类处理。
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 标签词典，可能为 ``None``
        """
        raise NotImplementedError('Tag vocabulary generation and checking '
                                  'method must be implemented. ')

    @abc.abstractmethod
    def on_dataset_preprocess(self, data_bundle: DataBundle,
                              tag_vocab: Optional[Vocabulary],
                              state_dict: Optional[dict]) -> DataBundle:
        """数据预处理，包括数据索引化和标签索引化。必须重写.

        :param data_bundle: 原始数据集，
            可能包含 ``train``、``dev``、``test``、``infer`` 四种，需要分类处理。
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 预处理后的数据集
        """
        raise NotImplementedError('Data preprocessing '
                                  'method must be implemented. ')

    @abc.abstractmethod
    def on_setup_model(self, data_bundle: DataBundle,
                       tag_vocab: Optional[Vocabulary],
                       state_dict: Optional[dict]):
        """模型构建，包括模型的初始化和加载。必须重写.

        :param data_bundle: 预处理后的数据集,
            可能包含 ``train``、``dev``、``test``、``infer`` 四种，需要分类处理。
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 拥有 ``train_step``、``evaluate_step``、
            ``inference_step`` 方法的对象
        """
        raise NotImplementedError('Model setup method must be implemented. ')

    @abc.abstractmethod
    def on_setup_optimizers(self, model, data_bundle: DataBundle,
                            tag_vocab: Optional[Vocabulary],
                            state_dict: Optional[dict]):
        """优化器构建，包括优化器的初始化和加载。必须重写.

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 优化器，可以为列表
        """
        raise NotImplementedError(
            'Optimizer setup method must be implemented. ')

    def on_setup_dataloader(self, model, data_bundle: DataBundle,
                            tag_vocab: Optional[Vocabulary],
                            state_dict: Optional[dict]) -> dict:
        """数据加载器构建，包括数据加载器的初始化和加载。不需要重写.

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 返回值为 ``dict`` 对象, 其中 key 为 ``train`` 的会被用作训练集,
            其他的会被根据当前控制器的 ``flag`` 选择而被设置为验证集, 测试机, 和推理集:

            例如:

            .. code-block:: python
                {
                    'train': train_dataloader,
                    'dev1': dev_dataloader,
                    'dev2': dev_dataloader,
                }


        """
        if get_flag() == 'train':
            if 'dev' in data_bundle.datasets.keys():
                return {
                    'train':
                    prepare_dataloader(data_bundle.datasets['train'],
                                       batch_size=self.batch_size,
                                       shuffle=True),
                    'dev':
                    prepare_dataloader(data_bundle.datasets['dev'],
                                       batch_size=self.batch_size,
                                       shuffle=False)
                }
            else:
                return {
                    'train':
                    prepare_dataloader(data_bundle.datasets['train'],
                                       batch_size=self.batch_size,
                                       shuffle=True)
                }
        elif get_flag() == 'eval':
            return {
                'test':
                prepare_dataloader(data_bundle.datasets['test'],
                                   batch_size=self.batch_size,
                                   shuffle=False)
            }
        elif get_flag() == 'infer':
            return {
                'infer':
                prepare_dataloader(data_bundle.datasets['infer'],
                                   batch_size=self.batch_size,
                                   shuffle=False)
            }
        return {}

    def on_setup_callbacks(self, model,
                           data_bundle: DataBundle,
                           tag_vocab: Optional[Vocabulary],
                           state_dict: Optional[dict]) \
            -> Union[Callback, Sequence[Callback]]:
        """FastNLP 回调参数构建，包括回调对象的初始化和加载。默认为空。

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: FastNLP 回调对象，可以为列表
        """
        return []

    def on_setup_metrics(self, model, data_bundle: DataBundle,
                         tag_vocab: Optional[Vocabulary],
                         state_dict: Optional[dict]) -> dict:
        """FastNLP 评价指标构建，包括评价指标对象的初始化和加载。默认为空。

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: dict 类型，其中 key 表示 monitor，value 表示一个 metric，例
            如 ``{"acc1": Accuracy(), "acc2": Accuracy()}``；

            目前我们支持的 ``metric`` 的种类有以下几种：

            1. fastNLP 的 ``metric``：详见 http://www.fastnlp.top/docs/fastNLP/master/api/core.html#metrics.
               Metric`；
            2. torchmetrics；
            3. allennlp.training.metrics；
            4. paddle.metric；
        """
        return {}

    def on_setup_extra_fastnlp_parameters(self, model, data_bundle: DataBundle,
                                          tag_vocab: Optional[Vocabulary],
                                          state_dict: Optional[dict]) -> dict:
        """其他 FastNLP 的可用参数.

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: dict 类型的参数列表
        """
        return {}

    def on_get_state_dict(self, model, data_bundle: DataBundle,
                          tag_vocab: Optional[Vocabulary]) -> dict:
        """获取 ``state_dict`` 用来保存，和其他方法参数中的 ``state_dict`` 一致。

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :return: 最新的 ``state_dict``
        """
        state_dict = {'model': model.state_dict()}
        if tag_vocab is not None:
            state_dict['tag_vocab'] = tag_vocab.word2idx
        return state_dict

    def _run_generator(self):

        def run(data_bundle: DataBundle):
            self.refresh_cache()

            def run_warp(data_bundle: DataBundle):
                parameters_or_data: dict = {}

                if self.load_model != '':
                    self._on_get_state_dict_cache = Hub.load(self.load_model)

                # 生命周期开始
                # 不易变的部分
                if not self._on_generate_and_check_tag_vocab_cache:
                    self._on_generate_and_check_tag_vocab_cache = \
                        self.on_generate_and_check_tag_vocab(
                            data_bundle=data_bundle,
                            state_dict=self._on_get_state_dict_cache)
                if not self._on_dataset_preprocess_cache:
                    self._on_dataset_preprocess_cache = \
                        self.on_dataset_preprocess(
                            data_bundle=data_bundle,
                            tag_vocab=self._on_generate_and_check_tag_vocab_cache,
                            state_dict=self._on_get_state_dict_cache)
                if not self._on_setup_model_cache:
                    self._on_setup_model_cache = \
                        self.on_setup_model(
                            data_bundle=self._on_dataset_preprocess_cache,
                            tag_vocab=self._on_generate_and_check_tag_vocab_cache,
                            state_dict=self._on_get_state_dict_cache)
                parameters_or_data['model'] = self._on_setup_model_cache
                if not hasattr(parameters_or_data['model'], 'train_step'):
                    logger.warning(
                        'The model you are using does not have a '
                        '`train_step` method, which is required for '
                        'training.')
                    if get_flag() == 'train':
                        raise RuntimeError('The model you are using does not '
                                           'have a `train_step` method, which '
                                           'is required for training.')
                if not hasattr(parameters_or_data['model'], 'evaluate_step'):
                    logger.warning(
                        'The model you are using does not have a '
                        '`evaluate_step` method, which is required for '
                        'evaluating.')
                    if get_flag() == 'eval':
                        raise RuntimeError(
                            'The model you are using does not '
                            'have a `evaluate_step` method, which '
                            'is required for evaluating.')
                if not hasattr(parameters_or_data['model'], 'inference_step'):
                    logger.warning(
                        'The model you are using does not have a '
                        '`inference_step` method, which is required for '
                        'infering.')
                    if get_flag() == 'infer' or get_flag() == 'interact':
                        raise RuntimeError(
                            'The model you are using does not '
                            'have a `inference_step` method, which '
                            'is required for inferring.')
                if get_flag() == 'train':
                    if not self._on_setup_optimizers_cache:
                        self._on_setup_optimizers_cache = \
                            self.on_setup_optimizers(
                                model=self._on_setup_model_cache,
                                data_bundle=self._on_dataset_preprocess_cache,
                                tag_vocab=self._on_generate_and_check_tag_vocab_cache,
                                state_dict=self._on_get_state_dict_cache)
                    parameters_or_data[
                        'optimizers'] = self._on_setup_optimizers_cache
                # 易变的部分
                # dataloader 部分
                self._on_setup_dataloader_cache = \
                    self.on_setup_dataloader(
                        model=self._on_setup_model_cache,
                        data_bundle=self._on_dataset_preprocess_cache,
                        tag_vocab=self._on_generate_and_check_tag_vocab_cache,
                        state_dict=self._on_get_state_dict_cache)
                if get_flag() == 'train':
                    if 'train' in self._on_setup_dataloader_cache.keys():
                        parameters_or_data['train_dataloader'] = \
                            self._on_setup_dataloader_cache.pop('train')
                if len(self._on_setup_dataloader_cache) == 1:
                    parameters_or_data['evaluate_dataloaders'] = \
                        next(iter(self._on_setup_dataloader_cache.values()))
                elif len(self._on_setup_dataloader_cache) > 1:
                    parameters_or_data['evaluate_dataloaders'] = \
                        self._on_setup_dataloader_cache
                # callbacks 部分
                self._on_setup_callbacks_cache = \
                    self.on_setup_callbacks(
                        model=self._on_setup_model_cache,
                        data_bundle=self._on_dataset_preprocess_cache,
                        tag_vocab=self._on_generate_and_check_tag_vocab_cache,
                        state_dict=self._on_get_state_dict_cache)
                if isinstance(self._on_setup_callbacks_cache, Sequence):
                    self._on_setup_callbacks_cache = \
                        [*self._on_setup_callbacks_cache]
                else:
                    self._on_setup_callbacks_cache = \
                        [self._on_setup_callbacks_cache]
                parameters_or_data['callbacks'] = \
                    self._on_setup_callbacks_cache
                # metric 部分
                self._on_setup_metrics_cache = \
                    self.on_setup_metrics(
                        model=self._on_setup_model_cache,
                        data_bundle=self._on_dataset_preprocess_cache,
                        tag_vocab=self._on_generate_and_check_tag_vocab_cache,
                        state_dict=self._on_get_state_dict_cache)
                parameters_or_data['metrics'] = self._on_setup_metrics_cache
                self._on_setup_extra_fastnlp_parameters_cache = \
                    self.on_setup_extra_fastnlp_parameters(
                        model=self._on_setup_model_cache,
                        data_bundle=self._on_dataset_preprocess_cache,
                        tag_vocab=self._on_generate_and_check_tag_vocab_cache,
                        state_dict=self._on_get_state_dict_cache)
                if self._on_setup_extra_fastnlp_parameters_cache is not None:
                    parameters_or_data.update(
                        self._on_setup_extra_fastnlp_parameters_cache)
                # 结束生命周期

                # device 相关参数
                if isinstance(self.device, str):
                    if ',' in self.device:
                        parameters_or_data['device'] = list(
                            map(lambda x: int(x), self.device.split(',')))
                    elif self.device.isdigit():
                        parameters_or_data['device'] = int(self.device)
                    else:
                        parameters_or_data['device'] = self.device
                else:
                    parameters_or_data['device'] = self.device
                # 保存模型相关
                if self.save_model_folder == '':
                    self.save_model_folder = os.getcwd()
                # monitor 相关
                if get_flag() == 'train' \
                        and (self.topk != 0 or self.load_best_model) \
                        and isinstance(self._on_setup_metrics_cache, dict) \
                        and len(self._on_setup_metrics_cache) > 0:
                    metrics_result = inspect_metrics(parameters_or_data)
                    if metrics_result:
                        if self.monitor == '':
                            logger.info(f'topk and load_best_model require '
                                        f'monitor to be set. The monitor that '
                                        f'can be selected include '
                                        f"[{','.join(metrics_result)}], "
                                        f'so {metrics_result[0]} will be set '
                                        f'as monitor')
                            self.monitor = metrics_result[0]
                        elif self.monitor not in metrics_result:
                            logger.warning(f'topk and load_best_model require '
                                           f'monitor to be set. The monitor '
                                           f'{self.monitor} you set is not in '
                                           f'the optional monitor range: '
                                           f"[{','.join(metrics_result)}], "
                                           f'so {metrics_result[0]} will be '
                                           f'set to monitor')
                        self.monitor = metrics_result[0]
                # is_large_better
                if isinstance(self.is_large_better, str):
                    if self.is_large_better.lower() == 'true':
                        self.is_large_better = True
                    elif self.is_large_better.lower() == 'false':
                        self.is_large_better = False
                    else:
                        raise ValueError(
                            f'`is_large_better` must be `True` or '
                            f'`False`, but got {self.is_large_better}')
                # topk 相关
                if self.topk != 0 \
                        and self.monitor != '' \
                        and get_flag() == 'train':

                    def model_save_fn(folder):
                        Hub.save(
                            os.path.join(folder, 'model.bin'),
                            self.on_get_state_dict(
                                model=self._on_setup_model_cache,
                                data_bundle=self._on_dataset_preprocess_cache,
                                tag_vocab=self.
                                _on_generate_and_check_tag_vocab_cache))

                    callback_parameters = dict(
                        folder=os.path.join(self.save_model_folder,
                                            'fastie_topk_model'),
                        topk=self.topk if self.topk != 0 else -self.topk,
                        larger_better=self.is_large_better,
                        model_save_fn=model_save_fn,
                        monitor=self.monitor)
                    callback = CheckpointCallback(**callback_parameters)
                    if self._on_setup_callbacks_cache is not None:
                        self._on_setup_callbacks_cache.append(callback)
                        parameters_or_data['callbacks'] = \
                            self._on_setup_callbacks_cache
                    else:
                        parameters_or_data['callbacks'] = [callback]
                # load_best_model 相关
                if self.load_best_model \
                        and isinstance(self._on_setup_metrics_cache, dict) \
                        and len(self._on_setup_metrics_cache) > 0 \
                        and get_flag() == 'train':

                    def model_save_fn(folder):
                        Hub.save(
                            os.path.join(folder, 'model.bin'),
                            self.on_get_state_dict(
                                model=self._on_setup_model_cache,
                                data_bundle=self._on_dataset_preprocess_cache,
                                tag_vocab=self.
                                _on_generate_and_check_tag_vocab_cache))

                    def model_load_fn(folder):
                        self._on_get_state_dict_cache = Hub.load(
                            os.path.join(folder, 'model.bin'))
                        self._on_setup_model_cache = None

                    callback = LoadBestModelCallback(
                        monitor=self.monitor,
                        model_save_fn=model_save_fn,
                        model_load_fn=model_load_fn,
                        save_folder=os.path.join(self.save_model_folder,
                                                 'fastie_best_model_cache'),
                        delete_after_train=False,
                        larger_better=self.is_large_better)
                    if self._on_setup_callbacks_cache is not None:
                        self._on_setup_callbacks_cache.append(callback)
                        parameters_or_data['callbacks'] = \
                            self._on_setup_callbacks_cache
                    else:
                        parameters_or_data['callbacks'] = [callback]
                # 训练轮数
                if get_flag() == 'train':
                    parameters_or_data['n_epochs'] = self.epochs
                # 混合精度
                parameters_or_data['fp16'] = self.fp16
                # 验证频率
                if get_flag() == 'train':
                    parameters_or_data['evaluate_every'] = self.evaluate_every
                # 为 infer 的情景修改执行函数
                if get_flag() == 'infer':
                    parameters_or_data['evaluate_fn'] = 'inference_step'
                return parameters_or_data

            if get_flag() is None:
                raise ValueError('You should set the flag first.')
            else:
                while True:
                    yield run_warp(data_bundle)

        return run

    def refresh_cache(self):
        self._on_generate_and_check_tag_vocab_cache = None
        self._on_dataset_preprocess_cache = None
        self._on_setup_model_cache = None
        self._on_setup_callbacks_cache = None
        self._on_setup_metrics_cache = None
        self._on_setup_extra_fastnlp_parameters_cache = None
        self._on_get_state_dict_cache = None

    def run(self, data_bundle: DataBundle):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if isinstance(self.run, Generator):
            return self.run
        else:
            return self.run(*args, **kwargs)
