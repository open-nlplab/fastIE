import abc
import os.path
from dataclasses import dataclass, field
from typing import Sequence, Union, Generator, Optional, Any

from fastNLP import Vocabulary, Callback, prepare_dataloader
from fastNLP.core.callbacks import CheckpointCallback
from fastNLP.io import DataBundle

from fastie.envs import get_flag, logger
from fastie.node import BaseNode, BaseNodeConfig
from fastie.utils.hub import Hub
from fastie.utils.registry import Registry
from fastie.utils.utils import generate_tag_vocab, check_loaded_tag_vocab

NER = Registry('NER')
RE = Registry('RE')
EE = Registry('EE')


@dataclass
class BaseTaskConfig(BaseNodeConfig, metaclass=abc.ABCMeta):
    cuda: bool = field(
        default=False,
        metadata=dict(
            help='Whether to use your NVIDIA graphics card to accelerate the '
            'process.',
            existence=True))
    load_model: str = field(
        default='',
        metadata=dict(help='Load the model from the path or model name. ',
                      existence=True))
    save_model: str = field(
        default='',
        metadata=dict(help='The path to save the model in last epoch '
                      '(Only available for train). ',
                      existence='train'))
    batch_size: int = field(default=32,
                            metadata=dict(help='Batch size. ', existence=True))
    shuffle: bool = field(default=True,
                          metadata=dict(
                              help='Whether to shuffle the dataset. ',
                              existence=True))
    epochs: int = field(default=20,
                        metadata=dict(help='Total number of training epochs. ',
                                      existence='train'))
    topk: int = field(default=0,
                      metadata=dict(
                          help='Save the top-k models according to metric. '
                          '(Only available for train). ',
                          existence='train'))
    monitor: str = field(default='',
                         metadata=dict(
                             help='The metric name which is used to select '
                             'the best model when using topk. '
                             'If this is not set, the first metric is used',
                             existence='train'))
    fp16: bool = field(default=False,
                       metadata=dict(help='Enable mixed-precision training. ',
                                     existence='train'))


class BaseTask(BaseNode):
    """FastIE 所有任务需要继承基类并重写生命周期方法。Task 的生命周期如下所示.

    （带星号号的为必须重写的方法,且标星号以及更上层的时间节点不会重复运行）：

        +-------+ state_dict/None           data_bundle
        |                +                       +
        |                |                       |
        |                |                       |
        |                v                       v
        |         +------+-----------------------+-------+
        |         |                                      |       tag_vocab
        |         |    on_generate_and_check_tag_vocab   +------+--------+
        |         |                                      |      |        |
        |         +------------------+-------------------+      |        |
        |                            |                          |        |
        |                            |  tag_vocab               |        |
        |                            v                          |        |
        |         +------------------+-------------------+      |        |
        |         |                                      |      |        |
        +-------> |         *on_dataset_preprocess       |      |        |
        |         |                                      |      |        |
        |         +------------------+-------------------+      |        |
        |                            |                          |        |
        |                            |  data_bundle             |        |
        |                            v                          |        |
        |         +------------------+-------------------+      |        |
        |         |                                      |  <---+        |
        +-------> |           *on_setup_model            |               |
        |         |                                      |  +---+        |
        |         +------------------+-------------------+      |        |
        |                            |                          |        |
        |                            |  model                   |        |
        |                            v                          |        |
        |         +------------------+-------------------+      |        |
        |         |                                      |      |        |
        +------>  |          *on_setup_optimizers        |      |        |
        |         |                                      |      |        |
        |         +--------------------------------------+      | model  |
        |                                                       |        |
        |                                                       |        |
        |                                                       |        |
        |         +--------------------------------------+      |        |
        |         |                                      |      |        |
        +-------> |           on_setup_dataloader        |      |        |
        |         |                                      |      |        |
        |         +--------------------------------------+      |        |
        |                                                       |        |
        |                                                       |        |
        |                                                       |        |
        |         +--------------------------------------+      |        |
        |         |                                      |      |        |
        +-------> |           on_setup_callbacks         |      |        |
        |         |                                      |      |        |
        |         +--------------------------------------+      |        |
        |                                                       |        |
        |                                                       |        |
        |                                                       |        |
        |         +--------------------------------------+      |        |
        |         |                                      |      |        |
        +-------> |            on_setup_metrics          |      |        |
        |         |                                      |      |        |
        |         +--------------------------------------+      |        |
        |                                                       |        |
        |                                                       |        |
        |                                                       |        |
        |         +--------------------------------------+      |        |
        |         |                                      |      |        |
        +-------> |   on_setup_extra_fastnlp_parameters  |      |        |
        |         |                                      |      |        |
        |         +--------------------------------------+      |        |
        |                                                       |        |
        |                                                       |        |
        |                                                       |        |
        |         +--------------------------------------+      |        |
        |         |                                      |  <---+        |
        +-------> |            on_get_state_dict         |               |
                  |                                      |  <------------+
                  +------------------+-------------------+
                                     |
                                     |
                                     |
                                     v
                                 state_dict

    :param cuda: 是否使用 GPU 加速训练
    :param load_model: 加载模型的路径或者模型名
    :param save_model: 训练结束后保存模型的路径
    :param batch_size: batch size
    :param epochs: 训练的轮数
    :param topk: 保存 metric 最好的 k 个模型
    :param monitor: 根据哪个 metric 选择  top-k 的模型
    :param fp16: 是否使用混合精度训练
    """
    _config = BaseTaskConfig()

    def __init__(self,
                 cuda: Union[bool, int, Sequence[int]] = False,
                 load_model: str = '',
                 save_model: str = '',
                 batch_size: int = 32,
                 shuffle: bool = True,
                 epochs: int = 20,
                 topk: int = 0,
                 monitor: str = '',
                 fp16: bool = False,
                 **kwargs):
        BaseNode.__init__(self, **kwargs)
        self.cuda = cuda
        self.load_model = load_model
        self.save_model = save_model
        self.epochs = epochs
        self.topk = topk
        self.monitor = monitor
        self.fp16 = fp16
        self.batch_size = batch_size
        self.shuffle = shuffle

        object.__setattr__(self, 'run', self._run_generator())

        self._on_generate_and_check_tag_vocab_cache: Any = None
        self._on_dataset_preprocess_cache: Any = None
        self._on_setup_model_cache: Any = None
        self._on_setup_optimizers_cache: Any = None
        self._on_setup_dataloader_cache: Any = None
        self._on_setup_callbacks_cache: Union[Callback, Sequence[Callback]] \
            = []
        self._on_setup_metrics_cache: Any = None
        self._on_setup_extra_fastnlp_parameters_cache: Any = None
        self._on_get_state_dict_cache: Any = None

    def on_generate_and_check_tag_vocab(self,
                                        data_bundle: DataBundle,
                                        state_dict: Optional[dict]) \
            -> Optional[Vocabulary]:
        """根据数据集中的已标注数据的标签词频生成标签词典。 如果加载模型得到的 ``state_dict`` 中存在
        ``tag_vocab``，则检查是否与 根据 ``data_bundle`` 生成的 tag_vocab 一致。

        :param data_bundle: 原始数据集，
            可能包含 ``train``、``dev``、``test``、``infer`` 四种，需要分类处理。
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 标签词典，可能为 ``None``
        """
        tag_vocab = None
        if state_dict is not None and 'tag_vocab' in state_dict:
            tag_vocab = state_dict['tag_vocab']
        signal, tag_vocab = check_loaded_tag_vocab(
            tag_vocab, generate_tag_vocab(data_bundle))
        if signal == -1:
            logger.warning(f'It is detected that the model label vocabulary '
                           f'conflicts with the dataset label vocabulary, '
                           f'so the model loading may fail. ')
        return tag_vocab

    @abc.abstractmethod
    def on_dataset_preprocess(self, data_bundle: DataBundle,
                              tag_vocab: Vocabulary,
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
    def on_setup_model(self, data_bundle: DataBundle, tag_vocab: Vocabulary,
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
                            tag_vocab: Vocabulary, state_dict: Optional[dict]):
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
                            tag_vocab: Vocabulary, state_dict: Optional[dict]):
        """数据加载器构建，包括数据加载器的初始化和加载。不需要重写.

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 返回值可用是可迭代的 dataloader，也可以返回多个 dataloader，
            当 flag 为 train 的时候返回的第一个 dataloader 会被用于训练，其他的用于验证
            当 flag 为 eval 或 infer 的时候都会被用于验证或推理，
        """
        if get_flag() == 'train':
            if 'dev' in data_bundle.datasets.keys():
                return prepare_dataloader(
                    data_bundle.datasets['train'],
                    batch_size=self.batch_size,
                    shuffle=self.shuffle), \
                    prepare_dataloader(data_bundle.datasets['dev'])
            else:
                return prepare_dataloader(data_bundle.datasets['train'],
                                          batch_size=self.batch_size,
                                          shuffle=self.shuffle)
        elif get_flag() == 'eval':
            return prepare_dataloader(data_bundle.datasets['dev'])
        elif get_flag() == 'infer' or get_flag() == 'interact':
            return prepare_dataloader(data_bundle.datasets['infer'])
    def on_setup_callbacks(self, model,
                           data_bundle: DataBundle,
                           tag_vocab: Vocabulary,
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
                         tag_vocab: Vocabulary,
                         state_dict: Optional[dict]) -> dict:
        """FastNLP 评价指标构建，包括评价指标对象的初始化和加载。默认为空。

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :param state_dict: 加载模型得到的 ``state_dict``，可能为 ``None``
        :return: dict 类型，其中 key 表示 monitor，value 表示一个 metric，例
            如 ``{"acc1": Accuracy(), "acc2": Accuracy()}``；

            目前我们支持的 ``metric`` 的种类有以下几种：

            1. fastNLP 的 ``metric``：详见 :class:`~fastNLP.core.metrics.
               Metric`；
            2. torchmetrics；
            3. allennlp.training.metrics；
            4. paddle.metric；
        """
        return {}

    def on_setup_extra_fastnlp_parameters(self, model, data_bundle: DataBundle,
                                          tag_vocab: Vocabulary,
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
                          tag_vocab: Vocabulary,
                          loaded_state_dict: Optional[dict]) -> dict:
        """获取 ``state_dict`` 用来保存，和其他方法参数中的 ``state_dict`` 一致。

        :param model: 初始化和加载后的模型
        :param data_bundle: 预处理后的数据集
        :param tag_vocab: 标签词典
        :param loaded_state_dict: 上次加载模型得到的 ``state_dict``，可能为 ``None``
        :return: 最新的 ``state_dict``
        """
        state_dict = {'model': model.state_dict()}
        if tag_vocab is not None:
            state_dict['tag_vocab'] = tag_vocab.word2idx
        return state_dict

    def _run_generator(self):

        def run(data_bundle: DataBundle):

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
                self._on_setup_dataloader_cache = \
                    self.on_setup_dataloader(
                        model=self._on_setup_model_cache,
                        data_bundle=self._on_dataset_preprocess_cache,
                        tag_vocab=self._on_generate_and_check_tag_vocab_cache,
                        state_dict=self._on_get_state_dict_cache)
                dataloaders = self._on_setup_dataloader_cache
                if isinstance(dataloaders, tuple):
                    if get_flag() == 'train':
                        parameters_or_data['train_dataloader'] = dataloaders[0]
                        if hasattr(parameters_or_data['model'],
                                   'evaluate_step'):
                            parameters_or_data['evaluate_dataloaders'] = \
                                {f'eval{i}': dataloaders[i]
                                 for i in range(1, len(dataloaders[1:]))}
                    else:
                        parameters_or_data['evaluate_dataloaders'] = \
                            {f'eval{i + 1}': dataloaders[i]
                             for i in range(len(dataloaders))}
                else:
                    if get_flag() == 'train':
                        parameters_or_data['train_dataloader'] = dataloaders
                    else:
                        parameters_or_data['evaluate_dataloaders'] = \
                            dataloaders
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

                # cuda 相关参数
                if isinstance(self.cuda, bool):
                    if self.cuda:
                        parameters_or_data['device'] = 0
                    else:
                        parameters_or_data['device'] = 'cpu'
                elif isinstance(self.cuda, Sequence) and isinstance(
                        self.cuda[0], int) or isinstance(self.cuda, int):
                    parameters_or_data['device'] = self.cuda
                # 保存模型参数
                if self.save_model != '':

                    def fastie_save_step():
                        if self._on_setup_model_cache is not None:
                            self._on_get_state_dict_cache \
                                = self.on_get_state_dict(
                                model=self._on_setup_model_cache,
                                data_bundle=self._on_dataset_preprocess_cache,
                                tag_vocab=
                                self._on_generate_and_check_tag_vocab_cache,
                                loaded_state_dict=
                                self._on_get_state_dict_cache)
                            Hub.save(self.save_model,
                                     self._on_get_state_dict_cache)

                    setattr(self._on_setup_model_cache, 'fastie_save_step',
                            fastie_save_step)
                # 不保存模型
                else:

                    def fastie_save_step():
                        pass

                    setattr(self._on_setup_model_cache, 'fastie_save_step',
                            fastie_save_step)
                # topk 相关
                if self.topk != 0 \
                        and isinstance(self._on_setup_metrics_cache, dict) \
                        and len(self._on_setup_metrics_cache) > 0:

                    def model_save_fn(folder):
                        self._on_get_state_dict_cache \
                            = self.on_get_state_dict(
                            model=self._on_setup_model_cache,
                            data_bundle=self._on_dataset_preprocess_cache,
                            tag_vocab=
                            self._on_generate_and_check_tag_vocab_cache,
                            loaded_state_dict=
                            self._on_get_state_dict_cache)
                        Hub.save(self.save_model,
                                 self._on_get_state_dict_cache)

                    callback = CheckpointCallback(
                        folder=os.getcwd(),
                        topk=self.topk if self.topk != 0 else -self.topk,
                        larger_better=(self.topk > 0),
                        model_save_fn=model_save_fn,
                        monitor=self.monitor if self.monitor != '' else list(
                            self._on_setup_metrics_cache.keys())[0])
                    if self._on_setup_callbacks_cache is not None:
                        self._on_setup_callbacks_cache.append(callback)
                        parameters_or_data['callbacks'] = \
                            self._on_setup_callbacks_cache
                    else:
                        parameters_or_data['callbacks'] = [callback]
                # 训练轮数
                parameters_or_data['n_epochs'] = self.epochs
                # 混合精度
                parameters_or_data['fp16'] = self.fp16
                # 为 infer 的情景修改执行函数
                if get_flag() == 'infer' or get_flag() == 'interact':
                    parameters_or_data['evaluate_fn'] = 'inference_step'
                return parameters_or_data

            if get_flag() is None:
                raise ValueError('You should set the flag first.')
            else:
                while True:
                    yield run_warp(data_bundle)

        return run

    def run(self, data_bundle: DataBundle):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if isinstance(self.run, Generator):
            return self.run
        else:
            return self.run(*args, **kwargs)
